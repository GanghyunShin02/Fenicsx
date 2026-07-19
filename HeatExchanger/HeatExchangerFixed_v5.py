import numpy as np
import matplotlib.pyplot as plt

import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx import fem
from dolfinx import mesh,io
from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio
from dolfinx.mesh import (locate_entities_boundary,
                            create_submesh)
import ufl
from ufl import (grad,
                 dot,
                 inner,
                 TrialFunction,
                 TestFunction,
                 dx, lhs,
                 nabla_grad,
                 div, rhs,
                 sym)
from dolfinx.fem import (Function, 
                         functionspace,
                         dirichletbc,
                         locate_dofs_topological,
                         form,
                         Constant,
                         extract_function_spaces)
from dolfinx.fem.petsc import (assemble_matrix,
                               assemble_vector,
                               apply_lifting, 
                               create_matrix, 
                               create_vector,
                               set_bc)

from CoolProp.CoolProp import PropsSI
from pathlib import Path

from dolfinx.fem.petsc import create_vector as create_vector_petsc
import inspect
import time
start=time.time()



gmsh.initialize()

r1, r2, r3, r4 = 1, 1.1, 2.1, 2.2
x0, y0, z0, L = 0, 0, 2.2, 10

if MPI.COMM_WORLD.rank == 0:


    c1 = gmsh.model.occ.addCylinder(x0, y0, z0, L, 0, 0, r1)
    c2 = gmsh.model.occ.addCylinder(x0, y0, z0, L, 0, 0, r2)
    c3 = gmsh.model.occ.addCylinder(x0, y0, z0, L, 0, 0, r3)
    c4 = gmsh.model.occ.addCylinder(x0, y0, z0, L, 0, 0, r4)

    c1_cp  = gmsh.model.occ.copy([(3, c1)])
    c2_cp1 = gmsh.model.occ.copy([(3, c2)])
    c2_cp2 = gmsh.model.occ.copy([(3, c2)])
    c3_cp1 = gmsh.model.occ.copy([(3, c3)])
    c3_cp2 = gmsh.model.occ.copy([(3, c3)])

    gmsh.model.occ.remove([(3, c2)])
    gmsh.model.occ.remove([(3, c3)])

    inpipe_dt,  _ = gmsh.model.occ.cut(c2_cp1, c1_cp)
    outflow_dt, _ = gmsh.model.occ.cut(c3_cp1, c2_cp2)
    outpipe_dt, _ = gmsh.model.occ.cut([(3, c4)], c3_cp2)
    inflow_dt = [(3, c1)]

    all_dt = inflow_dt + inpipe_dt + outflow_dt + outpipe_dt
    out, out_map = gmsh.model.occ.fragment(all_dt, [])
    gmsh.model.occ.synchronize()

    

    inflow  = [t for d, t in out_map[0]]
    inpipe  = [t for d, t in out_map[1]]
    outflow = [t for d, t in out_map[2]]
    outpipe = [t for d, t in out_map[3]]

    all_tags = inflow + inpipe + outflow + outpipe
    assert len(all_tags) == len(set(all_tags)), f"중복 태그: {all_tags}"
    total_volumes = len(gmsh.model.occ.getEntities(3))
    assert total_volumes == len(all_tags), f"미할당 volume 있음: total={total_volumes}, tagged={len(all_tags)}"
    inpipe_boundary = gmsh.model.getBoundary(
        [(3, t) for t in inpipe], oriented=False, combined=False
    )
    outpipe_boundary = gmsh.model.getBoundary(
        [(3, t) for t in outpipe], oriented=False, combined=False
    )

    inpipe_surf_tags = list(set(t for d, t in inpipe_boundary))
    outpipe_surf_tags = list(set(t for d, t in outpipe_boundary))

    gmsh.model.addPhysicalGroup(2, inpipe_surf_tags, 10, name="inpipe_walls")
    gmsh.model.addPhysicalGroup(2, outpipe_surf_tags, 11, name="outpipe_walls")



    gmsh.model.addPhysicalGroup(3, inflow, 1)
    gmsh.model.addPhysicalGroup(3, inpipe, 2)
    gmsh.model.addPhysicalGroup(3, outflow, 3)
    gmsh.model.addPhysicalGroup(3, outpipe, 4)

    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.1)
    gmsh.model.mesh.generate(3)


mesh_data=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,3)
domain=mesh_data.mesh

facet_marker=mesh_data.facet_tags
cell_marker=mesh_data.cell_tags

gmsh.finalize()

tdim = domain.topology.dim
fdim = tdim - 1

inflow_cells  = cell_marker.find(1)
outflow_cells = cell_marker.find(3)

inflow_submesh,  inflow_cell_map,  inflow_vertex_map,  _ = create_submesh(domain, tdim, inflow_cells)
outflow_submesh, outflow_cell_map, outflow_vertex_map, _ = create_submesh(domain, tdim, outflow_cells)


geom_dim = inflow_submesh.geometry.dim

Vu_in  = functionspace(inflow_submesh,  ("Lagrange", 2, (geom_dim,)))
VP_in  = functionspace(inflow_submesh,  ("Lagrange", 1))

Vu_out = functionspace(outflow_submesh, ("Lagrange", 2, (geom_dim,)))
VP_out = functionspace(outflow_submesh, ("Lagrange", 1))


u_zero = np.zeros(geom_dim, dtype=default_scalar_type)

x0, y0, z0 = 0, 0, 2.2
r1, r2, r3, r4 = 1, 1.1, 2.1, 2.2

def inflow_wall(x):
    r = np.sqrt((x[1]-y0)**2 + (x[2]-z0)**2)
    return np.isclose(r, r1, atol=1e-3)

def outflow_inner_wall(x):
    r = np.sqrt((x[1]-y0)**2 + (x[2]-z0)**2)
    return np.isclose(r, r2, atol=1e-3)

def outflow_outer_wall(x):
    r = np.sqrt((x[1]-y0)**2 + (x[2]-z0)**2)
    return np.isclose(r, r3, atol=1e-3)

outflow_inner_facets = locate_entities_boundary(outflow_submesh, fdim, outflow_inner_wall)
outflow_outer_facets = locate_entities_boundary(outflow_submesh, fdim, outflow_outer_wall)
outflow_wall_facets  = np.concatenate([outflow_inner_facets, outflow_outer_facets])

outflow_wall_dofs = locate_dofs_topological(Vu_out, fdim, outflow_wall_facets)
bc_outflow_wall = dirichletbc(u_zero, outflow_wall_dofs, Vu_out)

def inlet_face(x):
    return np.isclose(x[0], x0, atol=1e-6)

U_inlet = 1
u_inlet_val = np.array([U_inlet, 0.0, 0.0], dtype=default_scalar_type)

def outlet_face(x):
    return np.isclose(x[0], x0 + L, atol=1e-6)

outlet_facets = locate_entities_boundary(outflow_submesh, fdim, outlet_face)
outlet_dofs   = locate_dofs_topological(Vu_out, fdim, outlet_facets)

U_outlet = 10
# ===== [v2] annulus 유입속도(x=L, 사실상 inlet)도 램핑 =====
# 초기 u=0 정지 유체에 10 m/s를 즉시 걸면 압력보정 overshoot -> 첫 스텝부터 Un max 20+
u_annulus_in = Constant(outflow_submesh, np.array([0.0, 0.0, 0.0], dtype=default_scalar_type))
bc_outlet = dirichletbc(u_annulus_in, outlet_dofs, Vu_out)

def outlet_faceP(x):
    return np.isclose(x[0], x0 + L, atol=1e-6)

p_outlet_val = default_scalar_type(0.0)

def outflow_outlet_faceP(x):
    return np.isclose(x[0], x0, atol=1e-6)

outflow_outlet_facetsP = locate_entities_boundary(outflow_submesh, fdim, outflow_outlet_faceP)
outflow_outlet_dofsP = locate_dofs_topological(VP_out, fdim, outflow_outlet_facetsP)

P_outflow_outlet_val = default_scalar_type(0.0)
bc_outflow_outlet_P = dirichletbc(P_outflow_outlet_val, outflow_outlet_dofsP, VP_out)

V_T=functionspace(domain,("Lagrange",1))

Q_in=functionspace(inflow_submesh,("Lagrange",1))
Q_out=functionspace(outflow_submesh,("Lagrange",1))



T_init_inpipe_fluid  = 300.0
T_init_outpipe_fluid = 1000.0   # 정상 운전 시 열풍 온도 (BC 램핑 목표값, 물성 기준온도)
T_init_solid = 300.0
# ===== [v2] 콜드스타트: outflow 초기 온도장은 300K에서 시작 =====
# BC를 300->1000K로 램핑하므로 초기장도 300K여야 불연속이 안 생김
# (초기장 1000K + 램핑 BC 300K 조합은 반대방향 불연속을 만들어 overshoot 유발)
T_init_outflow_IC = 300.0

T_n = Function(V_T)


Q_T_in  = functionspace(inflow_submesh,  ("Lagrange", 1))
Q_T_out = functionspace(outflow_submesh, ("Lagrange", 1))

T_in_local  = Function(Q_T_in)
T_out_local = Function(Q_T_out)

T_in_local.x.array[:]  = T_init_inpipe_fluid
T_out_local.x.array[:] = T_init_outflow_IC   # [v2] 콜드스타트
T_in_local.x.scatter_forward()
T_out_local.x.scatter_forward()

# ===== [수정1-v2] 반응률(Arrhenius) 평가용 클리핑 온도 =====
# 물성은 상수로 풀므로 클리핑은 exp(-Ea/RT) 폭주 방지용으로만 사용
T_CLIP_MIN, T_CLIP_MAX = 250.0, 1250.0
T_in_clip  = Function(Q_T_in)
T_in_clip.x.array[:]  = T_init_inpipe_fluid
T_in_clip.x.scatter_forward()



inflow_cells  = cell_marker.find(1)
inpipe_cells  = cell_marker.find(2)
outflow_cells = cell_marker.find(3)
outpipe_cells = cell_marker.find(4)

inflow_dofs  = fem.locate_dofs_topological(V_T, tdim, inflow_cells)
inpipe_dofs  = fem.locate_dofs_topological(V_T, tdim, inpipe_cells)
outflow_dofs = fem.locate_dofs_topological(V_T, tdim, outflow_cells)
outpipe_dofs = fem.locate_dofs_topological(V_T, tdim, outpipe_cells)

T_n.x.array[inflow_dofs]  = T_init_inpipe_fluid
T_n.x.array[inpipe_dofs]  = T_init_solid
T_n.x.array[outflow_dofs] = T_init_outflow_IC   # [v2] 콜드스타트 (BC가 300->1000K 램핑)
T_n.x.array[outpipe_dofs] = T_init_solid

T_n.x.scatter_forward()


rho_h2_coeffs = [-1.99223776e-10, 5.07233264e-07, -4.63186950e-04, 1.78866855e-01]
cp_h2_coeffs  = [ 2.86578588e-06, -4.71896139e-03, 3.06441100e+00, 1.38033520e+04]
mu_h2_coeffs  = [ 2.80482839e-15, -9.23995890e-12, 2.49448659e-08, 2.22434806e-06]
k_h2_coeffs   = [ 1.46326830e-10, -3.15425362e-07, 5.97125665e-04, 3.29720370e-02]

def rho_h2(T):
    a, b, c, d = rho_h2_coeffs
    return a*T**3 + b*T**2 + c*T + d
    #return a*300**3 + b*300**2 + c*T + d

def cp_h2(T):
    a, b, c, d = cp_h2_coeffs
    return a*T**3 + b*T**2 + c*T + d

def mu_h2(T):
    a, b, c, d = mu_h2_coeffs
    return a*T**3 + b*T**2 + c*T + d

def k_h2(T):
    a, b, c, d = k_h2_coeffs
    return a*T**3 + b*T**2 + c*T + d

rho_air_coeffs = [-2.87425984e-09, 7.31461963e-06, -6.67479098e-03, 2.57469774e+00]
cp_air_coeffs  = [-3.75363227e-07, 8.16452532e-04, -3.48196493e-01, 1.04703821e+03]
mu_air_coeffs  = [ 1.07417126e-14, -3.37789360e-11, 6.43124367e-08, 2.03233228e-06]
k_air_coeffs   = [ 1.30177989e-11, -4.05872057e-08, 9.36400827e-05, 1.63976286e-03]

def rho_air(T):
    a, b, c, d = rho_air_coeffs
    return a*T**3 + b*T**2 + c*T + d

def cp_air(T):
    a, b, c, d = cp_air_coeffs
    return a*T**3 + b*T**2 + c*T + d

def mu_air(T):
    a, b, c, d = mu_air_coeffs
    return a*T**3 + b*T**2 + c*T + d

def k_air(T):
    a, b, c, d = k_air_coeffs
    return a*T**3 + b*T**2 + c*T + d



tstep = 0.001
dt_out = Constant(outflow_submesh, default_scalar_type(tstep))

# ===== [v2] 물성 전부 상수 (변밀도는 약형식 자체를 low-Mach로 바꿔야 하므로 보류) =====
T_ref_in  = 300.0    # H2 물성 기준온도
T_ref_out = 1000.0   # air 물성 기준온도
rho = Constant(inflow_submesh, default_scalar_type(rho_h2(T_ref_in)))
mu  = Constant(inflow_submesh, default_scalar_type(mu_h2(T_ref_in)))

U  = TrialFunction(Vu_out)
V  = TestFunction(Vu_out)
P  = TrialFunction(VP_out)
Q  = TestFunction(VP_out)
Un  = Function(Vu_out)
Un1 = Function(Vu_out)
Us  = Function(Vu_out)

RHO = Constant(outflow_submesh, default_scalar_type(rho_air(T_ref_out)))
MU  = Constant(outflow_submesh, default_scalar_type(0.2))

P_   = Function(VP_out)
PHI  = Function(VP_out)

def epsilon(v):
    return sym(grad(v))

dp = 0.003
pphi = 0.4
kappa = (dp**2 * pphi**3) / (150*(1-pphi)**2)     
betta = (1.75/dp) * ((1-pphi)/pphi**3)            

inflow_wall_facets = locate_entities_boundary(inflow_submesh, fdim, inflow_wall)
inflow_wall_dofs   = locate_dofs_topological(Vu_in, fdim, inflow_wall_facets)
bc_wall_darcy = dirichletbc(u_zero, inflow_wall_dofs, Vu_in)

inlet_facets = locate_entities_boundary(inflow_submesh, fdim, inlet_face)
inlet_dofs   = locate_dofs_topological(Vu_in, fdim, inlet_facets)
bc_inlet_darcy = dirichletbc(u_inlet_val, inlet_dofs, Vu_in)

outlet_facets_darcy = locate_entities_boundary(inflow_submesh, fdim, outlet_faceP)
outlet_p_dofs_darcy = locate_dofs_topological(VP_in, fdim, outlet_facets_darcy)
bc_outlet_p_darcy = dirichletbc(p_outlet_val, outlet_p_dofs_darcy, VP_in)

bcs_u_darcy = [bc_wall_darcy, bc_inlet_darcy]
bcs_p_darcy = [bc_outlet_p_darcy]

dp = 0.003
pphi = 0.4
kappa = (dp**2 * pphi**3) / (150*(1-pphi)**2)
betta = (1.75/dp) * ((1-pphi)/pphi**3)

kappa_constant = Constant(inflow_submesh, default_scalar_type(kappa))
betta_constant = Constant(inflow_submesh, default_scalar_type(betta))

u_d = TrialFunction(Vu_in)
v_d = TestFunction(Vu_in)
p_d = TrialFunction(VP_in)
q_d = TestFunction(VP_in)

u_old = Function(Vu_in)
u_new = Function(Vu_in)
p_darcy = Function(VP_in)
phi_darcy = Function(VP_in)

u_mag_old = ufl.sqrt(ufl.dot(u_old, u_old) + 1e-10)

F_u_darcy = (
    (mu/kappa_constant) * ufl.inner(u_d, v_d) * ufl.dx
    + (rho*betta_constant) * u_mag_old * ufl.inner(u_d, v_d) * ufl.dx
    - p_darcy * ufl.div(v_d) * ufl.dx
)
a_u_darcy = form(lhs(F_u_darcy))
l_u_darcy = form(rhs(F_u_darcy))
print(f'ludarcy{type(l_u_darcy)}')


A_u_darcy = create_matrix(a_u_darcy)
L_u_darcy = create_vector(extract_function_spaces(l_u_darcy))



a_p_darcy = form(dot(grad(q_d), grad(p_d))*dx)
l_p_darcy = form(-div(u_new)*q_d*dx)
A_p_darcy = assemble_matrix(a_p_darcy, bcs=bcs_p_darcy)
A_p_darcy.assemble()
L_p_darcy = create_vector(extract_function_spaces(l_p_darcy))

a_c_darcy = form(dot(u_d, v_d)*dx)
l_c_darcy = form(dot(u_new, v_d)*dx - dot(grad(phi_darcy), v_d)*dx)
A_c_darcy = assemble_matrix(a_c_darcy)
A_c_darcy.assemble()
L_c_darcy = create_vector(extract_function_spaces(l_c_darcy))

solver_u_darcy = PETSc.KSP().create(inflow_submesh.comm)
solver_u_darcy.setType(PETSc.KSP.Type.GMRES)
pc_u_darcy = solver_u_darcy.getPC()
pc_u_darcy.setType(PETSc.PC.Type.JACOBI)

solver_p_darcy = PETSc.KSP().create(inflow_submesh.comm)
solver_p_darcy.setOperators(A_p_darcy)
solver_p_darcy.setType(PETSc.KSP.Type.MINRES)
pc_p_darcy = solver_p_darcy.getPC()
pc_p_darcy.setType(PETSc.PC.Type.HYPRE)
pc_p_darcy.setHYPREType("boomeramg")

solver_c_darcy = PETSc.KSP().create(inflow_submesh.comm)
solver_c_darcy.setOperators(A_c_darcy)
solver_c_darcy.setType(PETSc.KSP.Type.CG)
pc_c_darcy = solver_c_darcy.getPC()
pc_c_darcy.setType(PETSc.PC.Type.SOR)

max_picard = 30
tol_picard = 1e-6

# ===== [변경 4] Darcy Picard 반복을 함수로 감싸서 매 스텝 재호출 가능하게 =====
def solve_darcy():
    for picard_iter in range(max_picard):
        A_u_darcy.zeroEntries()
        assemble_matrix(A_u_darcy, a_u_darcy, bcs=bcs_u_darcy)
        A_u_darcy.assemble()
        solver_u_darcy.setOperators(A_u_darcy)

        with L_u_darcy.localForm() as loc:
            loc.set(0)
        assemble_vector(L_u_darcy, l_u_darcy)
        apply_lifting(L_u_darcy, [a_u_darcy], [bcs_u_darcy])
        L_u_darcy.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L_u_darcy, bcs_u_darcy)
        solver_u_darcy.solve(L_u_darcy, u_new.x.petsc_vec)
        u_new.x.scatter_forward()

        with L_p_darcy.localForm() as loc:
            loc.set(0)
        assemble_vector(L_p_darcy, l_p_darcy)
        apply_lifting(L_p_darcy, [a_p_darcy], [bcs_p_darcy])
        L_p_darcy.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L_p_darcy, bcs_p_darcy)
        solver_p_darcy.solve(L_p_darcy, phi_darcy.x.petsc_vec)
        phi_darcy.x.scatter_forward()

        p_darcy.x.array[:] += phi_darcy.x.array
        p_darcy.x.scatter_forward()

        with L_c_darcy.localForm() as loc:
            loc.set(0)
        assemble_vector(L_c_darcy, l_c_darcy)
        L_c_darcy.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver_c_darcy.solve(L_c_darcy, u_new.x.petsc_vec)
        u_new.x.scatter_forward()

        diff_local = np.abs(u_new.x.array - u_old.x.array).max()
        diff = inflow_submesh.comm.allreduce(diff_local, op=MPI.MAX)
        #print(f"Picard iter {picard_iter}: diff={diff:.6e}")
        u_old.x.array[:] = u_new.x.array
        u_old.x.scatter_forward()

        if diff < tol_picard:
            print(f"Darcy Picard 수렴 (iter={picard_iter})")
            break

solve_darcy()

u_result = u_new
p_result = p_darcy

print("u_result min/max:", u_result.x.array.min(), u_result.x.array.max())

rho_iron = 7870.0
k_iron_coeffs  = [1.1e-4, -0.184, 125.5]
cp_iron_coeffs = [-3.33e-5, 0.4533, 314.0]

def k_iron(T):
    a, b, c = k_iron_coeffs
    return a*T**2 + b*T + c
def cp_iron(T):
    a, b, c = cp_iron_coeffs
    return a*T**2 + b*T + c

Q_dg = functionspace(domain, ("DG", 0))
rhocp_field = Function(Q_dg)
k_field = Function(Q_dg)

rhocp_field.x.array[inflow_cells]  = rho_h2(T_init_inpipe_fluid) * cp_h2(T_init_inpipe_fluid)
rhocp_field.x.array[inpipe_cells]  = rho_iron * cp_iron(T_init_solid)
rhocp_field.x.array[outflow_cells] = rho_air(T_init_outpipe_fluid) * cp_air(T_init_outpipe_fluid)
rhocp_field.x.array[outpipe_cells] = rho_iron * cp_iron(T_init_solid)

k_field.x.array[inflow_cells]  = k_h2(T_init_inpipe_fluid)
k_field.x.array[inpipe_cells]  = k_iron(T_init_solid)
k_field.x.array[outflow_cells] = k_air(T_init_outpipe_fluid)
k_field.x.array[outpipe_cells] = k_iron(T_init_solid)

rhocp_field.x.scatter_forward()
k_field.x.scatter_forward()

from dolfinx.fem import create_interpolation_data

Vu_full = functionspace(domain, ("Lagrange", 2, (geom_dim,)))
u_full = Function(Vu_full)

domain_cells_all = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)

# ===== [수정3] 보간을 영역별 부모 셀로 제한 =====
# 기존: domain_cells_all 전체에 대해 두 서브메쉬를 번갈아 보간 -> 서브메쉬 밖 점들이
# 외삽/미탐색에 의존, 인터페이스 근처 값 오염 위험. 각 영역 셀에만 쓰도록 제한.
inflow_parent_cells  = cell_marker.find(1).astype(np.int32)
outflow_parent_cells = cell_marker.find(3).astype(np.int32)

interp_data_u_in  = create_interpolation_data(Vu_full, Vu_in,  inflow_parent_cells)
interp_data_u_out = create_interpolation_data(Vu_full, Vu_out, outflow_parent_cells)

u_full.interpolate_nonmatching(u_result, inflow_parent_cells, interp_data_u_in)

# ===== [변경 3] T_out_local interpolate 준비 (기존에 없었음) =====
outflow_cells_local = np.arange(outflow_submesh.topology.index_map(tdim).size_local, dtype=np.int32)
interp_data_T_out = create_interpolation_data(Q_T_out, V_T, outflow_cells_local)

T = TrialFunction(V_T)
w = TestFunction(V_T)

dt_T = Constant(domain, default_scalar_type(tstep))
# ===== [수정4] Crank-Nicolson(0.5) -> Backward Euler(1.0) =====
# CN은 700K 초기 불연속에서 감쇠 없는 진동(300K 미만 undershoot)을 만듦
T_theta = T

dx_full = ufl.Measure("dx", domain=domain, subdomain_data=cell_marker)

FT = rhocp_field*inner((T-T_n)/dt_T, w)*dx_full
FT += k_field*inner(grad(T_theta), grad(w))*dx_full
FT += rhocp_field*inner(dot(u_full, grad(T_theta)), w)*dx_full(1)
FT += rhocp_field*inner(dot(u_full, grad(T_theta)), w)*dx_full(3)

# ===== [수정5] SUPG 안정화: 표준 tau + 영역 1,3 모두 적용 =====
# 기존 문제 (1) 영역 1(packed bed)은 이류항이 있는데 SUPG가 없었음
#          (2) tau = h/2|u| 는 시간항/확산항 스케일 누락 -> 벽 근처 저속영역 부정확
# 참고: P1 온도 + DG0 k에서 div(k*grad(T))는 셀 내부에서 항등 0이라 잔차에서 제외
h = ufl.CellDiameter(domain)
unorm = ufl.sqrt(ufl.dot(u_full, u_full) + 1e-12)
alpha_field = k_field / rhocp_field
# [v4] tau에서 (2/dt)^2 항 제거: dt=1e-3에서 2/dt=2000이 2u/h(~25)를 지배해
# tau ~ dt/2로 붕괴 -> SUPG가 사실상 꺼짐 (h/2u 대비 ~80배 과소). 이류/확산 항만 사용.
tau_supg = ((2.0*unorm/h)**2 + (4.0*alpha_field/h**2)**2)**(-0.5)
# [v5] 잔차에서 (T-T_n)/dt 항 제거: dt << tau 인 경우 SUPG의 시간잔차 커플링이
# 불안정을 유발함 (Bochev et al. small-time-step SUPG instability).
# 이류항만 남기면 tau*(u.grad w)(u.grad T)는 양의 준정부호 -> 무조건 안정.
res_T = rhocp_field*dot(u_full, grad(T_theta))
FT += tau_supg * dot(u_full, grad(w)) * res_T * dx_full(1)
FT += tau_supg * dot(u_full, grad(w)) * res_T * dx_full(3)
# ===== 여기까지 =====





def T_inflow_inlet_marker(x):
    r = np.sqrt((x[1]-y0)**2 + (x[2]-z0)**2)
    return np.isclose(x[0], x0, atol=1e-6) & (r < r1)

def T_outflow_inlet_marker(x):
    r = np.sqrt((x[1]-y0)**2 + (x[2]-z0)**2)
    return np.isclose(x[0], x0+L, atol=1e-6) & (r > r2) & (r < r3)

T_inflow_inlet_facets = locate_entities_boundary(domain, fdim, T_inflow_inlet_marker)
T_inflow_inlet_dofs = locate_dofs_topological(V_T, fdim, T_inflow_inlet_facets)
bc_T_inflow_inlet = dirichletbc(default_scalar_type(T_init_inpipe_fluid), T_inflow_inlet_dofs, V_T)

T_outflow_inlet_facets = locate_entities_boundary(domain, fdim, T_outflow_inlet_marker)
T_outflow_inlet_dofs = locate_dofs_topological(V_T, fdim, T_outflow_inlet_facets)
# ===== [수정6] 1000K inlet을 램핑 =====
# t=0에 벽(300K)과 1000K Dirichlet이 맞닿는 불연속이 undershoot의 씨앗
# 루프에서 T_hot.value를 300 -> 1000으로 t_ramp 동안 올림
T_hot = Constant(domain, default_scalar_type(300.0))
t_ramp = 0.5   # [s] 램핑 시간
bc_T_outflow_inlet = dirichletbc(T_hot, T_outflow_inlet_dofs, V_T)

bcs_T = [bc_T_inflow_inlet, bc_T_outflow_inlet]


rho_h2_300 = rho_h2(300)
A_inlet = np.pi * r1**2
U_inlet = 1.0

m_dot = rho_h2_300 * A_inlet * U_inlet
M_H2 = 0.002016

n_dot_total = m_dot / M_H2

x_CO2 = 1/5
x_H2  = 4/5

n_dot_CO2 = x_CO2 * n_dot_total
n_dot_H2  = x_H2  * n_dot_total

Q_vol = A_inlet * U_inlet

C_CO2_init = n_dot_CO2 / Q_vol
C_H2_init  = n_dot_H2  / Q_vol
C_CH4_init = 0.0
C_H2O_init = 0.0

A_pre = 1.0e2
Ea = 66100.0
R_gas = 8.314
n_H2 = 0.88
n_CO2 = 0.34

def rate_constant(T):
    return A_pre * ufl.exp(-Ea/(R_gas*T))

def reaction_rate(T, C_H2, C_CO2):
    C_H2_safe = ufl.max_value(C_H2, 1e-10)
    C_CO2_safe = ufl.max_value(C_CO2, 1e-10)
    return rate_constant(T) * C_H2_safe**n_H2 * C_CO2_safe**n_CO2

Q_C = functionspace(inflow_submesh, ("Lagrange", 1))

C_CO2_trial = TrialFunction(Q_C)
w_C = TestFunction(Q_C)

C_CO2_n = Function(Q_C)
C_H2_n  = Function(Q_C)
C_CH4_n = Function(Q_C)
C_H2O_n = Function(Q_C)

C_CO2_new = Function(Q_C)
C_H2_new  = Function(Q_C)
C_CH4_new = Function(Q_C)
C_H2O_new = Function(Q_C)

C_CO2_n.x.array[:] = C_CO2_init
C_H2_n.x.array[:]  = C_H2_init
C_CH4_n.x.array[:] = 0.0
C_H2O_n.x.array[:] = 0.0

D_CO2 = Constant(inflow_submesh, default_scalar_type(2.0e-5))
D_H2  = Constant(inflow_submesh, default_scalar_type(8.0e-5))
D_CH4 = Constant(inflow_submesh, default_scalar_type(3.0e-5))
D_H2O = Constant(inflow_submesh, default_scalar_type(4.0e-5))

dt_C = Constant(inflow_submesh, default_scalar_type(tstep))

C_trial = TrialFunction(Q_C)
w_C = TestFunction(Q_C)

# ===== [수정7] species 이송방정식 전면 수정 =====
# 기존 문제 (1) SUPG 없음: 셀 Peclet ~10^3 -> 진동 -> outlet 농도 > 초기값 -> 전화율 음수
#          (2) 소모항이 완전 explicit: dt*rate > C_n 이면 농도가 음수로 뚫림
#          (3) 반응률에 T_in_local 직접 사용: T 진동 시 exp(-Ea/RT) 폭주 가능
# 반응률은 클리핑 온도(T_in_clip)로 평가
r_expr = reaction_rate(T_in_clip, C_H2_n, C_CO2_n)

k_rate    = rate_constant(T_in_clip)
C_H2_sf   = ufl.max_value(C_H2_n, 1e-10)
C_CO2_sf  = ufl.max_value(C_CO2_n, 1e-10)
# 소모종은 자기 농도에 대해 semi-implicit 선형화: r ≈ lin_i * C_trial
# -> LHS로 이동하면 대각 강화 + 양수성 보장 방향
lin_CO2 = k_rate * C_H2_sf**n_H2  * C_CO2_sf**(n_CO2 - 1.0)   # CO2 소모 (nu=1)
lin_H2  = k_rate * C_CO2_sf**n_CO2 * C_H2_sf**(n_H2 - 1.0)    # H2 소모 (nu=4)

h_in     = ufl.CellDiameter(inflow_submesh)
unorm_in = ufl.sqrt(dot(u_result, u_result) + 1e-12)

def species_form(C_trial, C_n, D_i, w_C, source=None, sink_lin=None):
    # [v4] tau에서 시간항 제거 (온도 SUPG와 동일한 이유)
    tau_C = ((2.0*unorm_in/h_in)**2 + (4.0*D_i/h_in**2)**2)**(-0.5)
    # [v5] 잔차에서도 (C-C_n)/dt 제거 (small-time-step SUPG 불안정 방지)
    res = dot(u_result, grad(C_trial))
    F = ((C_trial - C_n)/dt_C * w_C * ufl.dx
         + dot(u_result, grad(C_trial)) * w_C * ufl.dx
         + D_i * dot(grad(C_trial), grad(w_C)) * ufl.dx)
    if source is not None:          # 생성종: explicit 소스 (항상 양수라 안전)
        F   -= source * w_C * ufl.dx
        res -= source
    if sink_lin is not None:        # 소모종: semi-implicit 싱크
        F   += sink_lin * C_trial * w_C * ufl.dx
        res += sink_lin * C_trial
    F += tau_C * dot(u_result, grad(w_C)) * res * ufl.dx   # SUPG
    return F

F_CO2 = species_form(C_trial, C_CO2_n, D_CO2, w_C, sink_lin=1.0*lin_CO2)
F_H2  = species_form(C_trial, C_H2_n,  D_H2,  w_C, sink_lin=4.0*lin_H2)
F_CH4 = species_form(C_trial, C_CH4_n, D_CH4, w_C, source=1.0*r_expr)
F_H2O = species_form(C_trial, C_H2O_n, D_H2O, w_C, source=2.0*r_expr)

C_CO2_inlet_dofs = locate_dofs_topological(Q_C, fdim, inlet_facets)
bc_C_CO2_inlet = dirichletbc(default_scalar_type(C_CO2_init), C_CO2_inlet_dofs, Q_C)

C_H2_inlet_dofs = locate_dofs_topological(Q_C, fdim, inlet_facets)
bc_C_H2_inlet = dirichletbc(default_scalar_type(C_H2_init), C_H2_inlet_dofs, Q_C)

bc_C_CH4_inlet = dirichletbc(default_scalar_type(0.0), C_CO2_inlet_dofs, Q_C)
bc_C_H2O_inlet = dirichletbc(default_scalar_type(0.0), C_CO2_inlet_dofs, Q_C)

a_CO2 = form(lhs(F_CO2)); l_CO2 = form(rhs(F_CO2))
a_H2  = form(lhs(F_H2));  l_H2  = form(rhs(F_H2))
a_CH4 = form(lhs(F_CH4)); l_CH4 = form(rhs(F_CH4))
a_H2O = form(lhs(F_H2O)); l_H2O = form(rhs(F_H2O))

A_CO2 = create_matrix(a_CO2); L_CO2 = create_vector(extract_function_spaces(l_CO2))
A_H2  = create_matrix(a_H2);  L_H2  = create_vector(extract_function_spaces(l_H2))
A_CH4 = create_matrix(a_CH4); L_CH4 = create_vector(extract_function_spaces(l_CH4))
A_H2O = create_matrix(a_H2O); L_H2O = create_vector(extract_function_spaces(l_H2O))

def make_solver(A):
    solver = PETSc.KSP().create(inflow_submesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    return solver

solver_CO2 = make_solver(A_CO2)
solver_H2  = make_solver(A_H2)
solver_CH4 = make_solver(A_CH4)
solver_H2O = make_solver(A_H2O)




aT = form(lhs(FT))
lT = form(rhs(FT))
AT = create_matrix(aT)
LT = create_vector(extract_function_spaces(lT))

solverT = PETSc.KSP().create(domain.comm)
solverT.setType(PETSc.KSP.Type.GMRES)
pcT = solverT.getPC()
pcT.setType(PETSc.PC.Type.LU)
# pcT.setHYPREType("boomeramg")


inflow_cells_local = np.arange(inflow_submesh.topology.index_map(tdim).size_local, dtype=np.int32)
interp_data_T_in = create_interpolation_data(Q_T_in, V_T, inflow_cells_local)

Q_r_in = functionspace(inflow_submesh, ("Lagrange", 1))
r_field_local = Function(Q_r_in)
r_field = Function(V_T)
# ===== [수정8] r_field 보간도 inflow 영역 셀로 제한 =====
interp_data_r_in = create_interpolation_data(V_T, Q_r_in, inflow_parent_cells)
# 루프에서 r_field_local 갱신에 쓸 반응률 Expression (매번 새로 만들지 않도록 미리 컴파일)
r_rate_expr_compiled = fem.Expression(reaction_rate(T_in_clip, C_H2_n, C_CO2_n),
                                      Q_r_in.element.interpolation_points)

dHrxn = -165000.0
# ===== [수정15] 반응열 부호 수정 =====
# 열원 S = (-dHrxn)*r > 0 (발열). 약형식 잔차 F에는 -S*w = +dHrxn*r*w 로 들어가야 함.
# 기존 (- dHrxn * r * w) 부호면 발열반응이 오히려 온도를 낮추는 방향이었음
# (지금까진 r_field가 항상 0이라 드러나지 않았을 뿐)
FT_with_reaction = FT + dHrxn * r_field * w * dx_full(1)

aT2 = form(lhs(FT_with_reaction))
lT2 = form(rhs(FT_with_reaction))
AT2 = create_matrix(aT2)
LT2 = create_vector(extract_function_spaces(lT2))
solverT.setOperators(AT2)

outlet_marker_C = lambda x: np.isclose(x[0], x0+L, atol=1e-6)
outlet_facets_C = locate_entities_boundary(inflow_submesh, fdim, outlet_marker_C)
outlet_dofs_C = locate_dofs_topological(Q_C, fdim, outlet_facets_C)

history_t = []
history_X_CO2 = []
history_T_outlet = []
history_rate = []


F11 = (RHO/dt_out)*dot((U-Un), V)*dx
F11 += RHO*inner(dot((1.5*Un-0.5*Un1), (0.5*nabla_grad(U+Un))), V)*dx
F11 += 2*0.5*MU*inner(epsilon(V), epsilon(U+Un))*dx
F11 -= dot(P_, div(V))*dx

a11 = form(lhs(F11))
l11 = form(rhs(F11))
A11 = create_matrix(a11)
L11 = create_vector(extract_function_spaces(l11))

a22 = form((1/RHO)*dot(grad(Q), grad(P))*dx)
l22 = form(-(1.0/dt_out)*dot(div(Us), Q)*dx)
A22 = assemble_matrix(a22, bcs=[bc_outflow_outlet_P])
A22.assemble()
L22 = create_vector(extract_function_spaces(l22))

a33 = form(RHO*dot(U, V)*dx)
l33 = form(RHO*dot(Us, V)*dx - dt_out*dot(nabla_grad(PHI), V)*dx)
A33 = assemble_matrix(a33)
A33.assemble()
L33 = create_vector(extract_function_spaces(l33))








solver11 = PETSc.KSP().create(outflow_submesh.comm)
solver11.setOperators(A11)
solver11.setType(PETSc.KSP.Type.GMRES)
pc11 = solver11.getPC()
pc11.setType(PETSc.PC.Type.JACOBI)

solver22 = PETSc.KSP().create(outflow_submesh.comm)
solver22.setOperators(A22)
solver22.setType(PETSc.KSP.Type.MINRES)
pc22 = solver22.getPC()
pc22.setType(PETSc.PC.Type.HYPRE)
pc22.setHYPREType("boomeramg")

solver33 = PETSc.KSP().create(outflow_submesh.comm)
solver33.setOperators(A33)
solver33.setType(PETSc.KSP.Type.CG)
pc33 = solver33.getPC()
pc33.setType(PETSc.PC.Type.SOR)

vtx_u_in  = VTXWriter(inflow_submesh.comm,  "results/10u_inflow_darcy.bp", [u_result], engine="BP4")
vtx_p_in  = VTXWriter(inflow_submesh.comm,  "results/10p_inflow_darcy.bp", [p_result], engine="BP4")
vtx_u_out = VTXWriter(outflow_submesh.comm, "results/10u_outflow.bp",      [Un],       engine="BP4")
vtx_p_out = VTXWriter(outflow_submesh.comm, "results/10p_outflow.bp",      [P_],       engine="BP4")
vtx_T     = VTXWriter(domain.comm,          "results/10T.bp",              [T_n],      engine="BP4")

vtx_C_CO2 = VTXWriter(inflow_submesh.comm, "results/10C_CO2.bp", [C_CO2_n], engine="BP4")
vtx_C_H2  = VTXWriter(inflow_submesh.comm, "results/10C_H2.bp",  [C_H2_n],  engine="BP4")
vtx_C_CH4 = VTXWriter(inflow_submesh.comm, "results/10C_CH4.bp", [C_CH4_n], engine="BP4")
vtx_C_H2O = VTXWriter(inflow_submesh.comm, "results/10C_H2O.bp", [C_H2O_n], engine="BP4")


vtx_u_in.write(0.0)
vtx_p_in.write(0.0)
vtx_T.write(0.0)

t = 0.0
T_end = 10
num_steps = int(T_end / tstep)



print("T_outflow_inlet_dofs count:", len(T_outflow_inlet_dofs))

# ===== [수정14] outlet 온도 모니터링 dof: 루프 밖에서 한 번만 탐색 =====
T_outlet_dofs_loop = locate_dofs_topological(V_T, fdim,
    locate_entities_boundary(domain, fdim,
        lambda x: np.isclose(x[0], x0+L, atol=1e-6) & (np.sqrt((x[1]-y0)**2+(x[2]-z0)**2) < r1)))

# ===== [v5] MPI-safe 모니터링 =====
# 기존 진단은 rank 0 로컬 배열만 봄 -> mpirun 실행 시 outlet dof가 rank 0에 없으면
# X_CO2=nan, T_outlet=nan, min/max도 부분값만 출력됨. 전 랭크 reduce로 수정.
comm = MPI.COMM_WORLD
n_own_T  = V_T.dofmap.index_map.size_local
n_own_C  = Q_C.dofmap.index_map.size_local
n_own_To = Q_T_out.dofmap.index_map.size_local
n_own_r  = Q_r_in.dofmap.index_map.size_local
n_own_u3 = Vu_out.dofmap.index_map.size_local * Vu_out.dofmap.index_map_bs

outlet_dofs_C_own = outlet_dofs_C[outlet_dofs_C < n_own_C]
T_outlet_dofs_own = T_outlet_dofs_loop[T_outlet_dofs_loop < n_own_T]

def gmin(a): return comm.allreduce(float(a.min()) if a.size else np.inf,  op=MPI.MIN)
def gmax(a): return comm.allreduce(float(a.max()) if a.size else -np.inf, op=MPI.MAX)
def gavg(a):
    s = comm.allreduce(float(a.sum()) if a.size else 0.0, op=MPI.SUM)
    n = comm.allreduce(int(a.size), op=MPI.SUM)
    return s/n if n > 0 else float('nan')


for step in range(num_steps):
    t += tstep
    if MPI.COMM_WORLD.rank==0: 
        print(f'{100*step/num_steps:.1f}% 완료, t={t:.4f}')

    # ===== [수정9-v3] 램핑: 온도 300->1000K, annulus 속도 0->10 m/s =====
    ramp = min(t / t_ramp, 1.0)
    T_hot.value = 300.0 + 700.0 * ramp
    u_annulus_in.value[0] = -U_outlet * ramp

    # ===== [v3] 시간변화 Dirichlet + consistent mass 아티팩트 제거 =====
    # BC 값이 스텝마다 오르면 lifting의 -(M_ij/dt)*(g_new - T_n_j) 항이
    # 이웃 노드를 램핑 속도에 비례해 끌어내림 (관측: T_min = 300 - 0.40*(T_hot-300)).
    # 이전 해의 BC dof를 새 BC 값으로 미리 맞춰 (g_new - T_n_j)=0 으로 만들어 소멸시킴.
    T_n.x.array[T_outflow_inlet_dofs] = T_hot.value
    T_n.x.scatter_forward()
    Un.x.array[3*outlet_dofs]     = u_annulus_in.value[0]   # 속도도 동일 메커니즘
    Un.x.array[3*outlet_dofs + 1] = 0.0
    Un.x.array[3*outlet_dofs + 2] = 0.0
    Un.x.scatter_forward()
    Un1.x.array[3*outlet_dofs]     = u_annulus_in.value[0]
    Un1.x.array[3*outlet_dofs + 1] = 0.0
    Un1.x.array[3*outlet_dofs + 2] = 0.0
    Un1.x.scatter_forward()

    A11.zeroEntries()
    assemble_matrix(A11, a11, bcs=[bc_outflow_wall, bc_outlet])
    A11.assemble()

    with L11.localForm() as loc:
        loc.set(0)
    assemble_vector(L11, l11)
    apply_lifting(L11, [a11], [[bc_outflow_wall, bc_outlet]])
    L11.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L11, [bc_outflow_wall, bc_outlet])
    solver11.solve(L11, Us.x.petsc_vec)
    Us.x.scatter_forward()

    # [v2] RHO 상수이므로 A22 재조립 불필요 (루프 밖에서 1회 조립)
    with L22.localForm() as loc:
        loc.set(0)
    assemble_vector(L22, l22)
    apply_lifting(L22, [a22], [[bc_outflow_outlet_P]])
    L22.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L22, [bc_outflow_outlet_P])
    solver22.solve(L22, PHI.x.petsc_vec)
    PHI.x.scatter_forward()

    P_.x.array[:] += PHI.x.array
    P_.x.scatter_forward()

    # [v2] RHO 상수이므로 A33 재조립 불필요
    with L33.localForm() as loc:
        loc.set(0)
    assemble_vector(L33, l33)
    L33.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver33.solve(L33, Un.x.petsc_vec)
    Un.x.scatter_forward()

    Un1.x.array[:] = Un.x.array

    # ===== [수정10-a] outflow 속도 보간: outflow 부모 셀에만 =====
    u_full.interpolate_nonmatching(Un, outflow_parent_cells, interp_data_u_out)
    u_full.x.scatter_forward()

    T_out_local.interpolate_nonmatching(T_n, outflow_cells_local, interp_data_T_out)
    T_out_local.x.scatter_forward()

    # ===== [수정10-b v2] 물성은 상수. 클리핑은 반응률(Arrhenius) 보호용으로만 =====
    T_in_clip.x.array[:] = np.clip(T_in_local.x.array, T_CLIP_MIN, T_CLIP_MAX)
    T_in_clip.x.scatter_forward()

    # ===== [v5] 최저온도: 전 랭크 MINLOC =====
    To_own = T_out_local.x.array[:n_own_To]
    loc_min = float(To_own.min()) if To_own.size else np.inf
    glob_min, owner_rank = comm.allreduce((loc_min, comm.rank), op=MPI.MINLOC)
    if comm.rank == owner_rank:
        idx = int(np.argmin(To_own))
        print("최저온도 위치:", Q_T_out.tabulate_dof_coordinates()[idx],
              " 값:", glob_min, flush=True)

    solve_darcy()
    # ===== [수정10-c] inflow 속도 보간: inflow 부모 셀에만 =====
    u_full.interpolate_nonmatching(u_result, inflow_parent_cells, interp_data_u_in)
    u_full.x.scatter_forward()

    # ===== [수정11] species 4종 solve (기존엔 루프에서 아예 안 풀리고 있었음!) =====
    # 행렬은 u_result(darcy), C_n(선형화 계수), T_in_clip(반응률)에 의존 -> 매 스텝 재조립
    for A_i, a_i, L_i, l_i, s_i, bc_i, C_new_i in [
        (A_CO2, a_CO2, L_CO2, l_CO2, solver_CO2, bc_C_CO2_inlet, C_CO2_new),
        (A_H2,  a_H2,  L_H2,  l_H2,  solver_H2,  bc_C_H2_inlet,  C_H2_new),
        (A_CH4, a_CH4, L_CH4, l_CH4, solver_CH4, bc_C_CH4_inlet, C_CH4_new),
        (A_H2O, a_H2O, L_H2O, l_H2O, solver_H2O, bc_C_H2O_inlet, C_H2O_new),
    ]:
        A_i.zeroEntries()
        assemble_matrix(A_i, a_i, bcs=[bc_i])
        A_i.assemble()
        s_i.setOperators(A_i)
        with L_i.localForm() as loc:
            loc.set(0)
        assemble_vector(L_i, l_i)
        apply_lifting(L_i, [a_i], [[bc_i]])
        L_i.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L_i, [bc_i])
        s_i.solve(L_i, C_new_i.x.petsc_vec)
        C_new_i.x.scatter_forward()

    # new -> n 복사 + 음수 클리핑 (SUPG가 진동을 줄여도 완전 단조는 아니므로 안전장치)
    for C_new_i, C_n_i in [(C_CO2_new, C_CO2_n), (C_H2_new, C_H2_n),
                           (C_CH4_new, C_CH4_n), (C_H2O_new, C_H2O_n)]:
        C_n_i.x.array[:] = np.maximum(C_new_i.x.array, 0.0)
        C_n_i.x.scatter_forward()

    # ===== [수정12] 반응률 필드 갱신 (기존엔 영원히 0이었음 -> 반응열/Levenspiel 전부 무의미) =====
    r_field_local.interpolate(r_rate_expr_compiled)
    r_field_local.x.scatter_forward()
    r_field.interpolate_nonmatching(r_field_local, inflow_parent_cells, interp_data_r_in)
    r_field.x.scatter_forward()

    # ===== [수정13] 온도: 반응열 포함된 aT2/lT2/AT2 사용 =====
    # (기존엔 aT/AT를 조립해서 setOperators(AT2)를 덮어써버려 반응열이 전혀 반영 안 됐음)
    AT2.zeroEntries()
    assemble_matrix(AT2, aT2, bcs=bcs_T)
    AT2.assemble()
    solverT.setOperators(AT2)

    with LT2.localForm() as loc:
        loc.set(0)
    assemble_vector(LT2, lT2)
    apply_lifting(LT2, [aT2], [bcs_T])
    LT2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(LT2, bcs_T)
    solverT.solve(LT2, T_n.x.petsc_vec)
    T_n.x.scatter_forward()

    T_in_local.interpolate_nonmatching(T_n, inflow_cells_local, interp_data_T_in)
    T_in_local.x.scatter_forward()


    # ===== [v5] 모니터링 전부 전역 reduce (mpirun에서도 올바른 값) =====
    C_CO2_outlet_avg = gavg(C_CO2_n.x.array[outlet_dofs_C_own])
    X_CO2_now = (C_CO2_init - C_CO2_outlet_avg) / C_CO2_init

    T_outlet_avg = gavg(T_n.x.array[T_outlet_dofs_own])
    rate_avg = gavg(r_field_local.x.array[:n_own_r])

    Tn_min,  Tn_max = gmin(T_n.x.array[:n_own_T]),  gmax(T_n.x.array[:n_own_T])
    To_min,  To_max = gmin(T_out_local.x.array[:n_own_To]), gmax(T_out_local.x.array[:n_own_To])
    Un_min,  Un_max = gmin(Un.x.array[:n_own_u3]),  gmax(Un.x.array[:n_own_u3])
    C_min,   C_max  = gmin(C_CO2_n.x.array[:n_own_C]), gmax(C_CO2_n.x.array[:n_own_C])
    nan_any = comm.allreduce(bool(np.isnan(Un.x.array).any() or np.isnan(T_n.x.array).any()),
                             op=MPI.LOR)

    history_t.append(t)
    history_X_CO2.append(X_CO2_now)
    history_T_outlet.append(T_outlet_avg)
    history_rate.append(rate_avg)

    if comm.rank==0 and step%100==0:
        print(f"X_CO2={X_CO2_now:.4f}  T_outlet={T_outlet_avg:.2f}")
        print("T_out_local min/max:", To_min, To_max)
        print("Un min/max:", Un_min, Un_max)
        print("T_n min/max:", Tn_min, Tn_max)
        print("C_CO2 min/max:", C_min, C_max)
        print("rate avg:", rate_avg)
        print("NaN 감지:", nan_any)



        vtx_u_out.write(t)
        vtx_p_out.write(t)
        vtx_T.write(t)



vtx_u_in.close()
vtx_p_in.close()
vtx_u_out.close()
vtx_p_out.close()
vtx_T.close()

if MPI.COMM_WORLD.rank==0:
    print("inflow cells:", inflow_submesh.topology.index_map(inflow_submesh.topology.dim).size_global)

    T_dof_coords = V_T.tabulate_dof_coordinates()

    def get_avg_T_at_x_bin(x_target, x_tol, r_min, r_max):
        x_coords = T_dof_coords[:, 0]
        y_coords = T_dof_coords[:, 1]
        z_coords = T_dof_coords[:, 2]
        r = np.sqrt((y_coords - y0)**2 + (z_coords - z0)**2)

        mask = (np.abs(x_coords - x_target) < x_tol) & (r >= r_min) & (r <= r_max)
        if mask.sum() == 0:
            return np.nan
        return T_n.x.array[mask].mean()

    n_points = 50
    x_samples = np.linspace(x0, x0+L, n_points)
    x_tol = L / n_points

    T_inflow_profile  = [get_avg_T_at_x_bin(x, x_tol, 0, r1*0.9) for x in x_samples]
    T_outflow_profile = [get_avg_T_at_x_bin(x, x_tol, r2*1.05, r3*0.95) for x in x_samples]

    plt.figure(figsize=(8,5))
    plt.plot(x_samples, T_inflow_profile, 'r-', linewidth=2, label='Inner tube fluid (H2)')
    plt.plot(x_samples, T_outflow_profile, 'b-', linewidth=2, label='Outer tube fluid (Air)')
    plt.xlabel('Tube length (m)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Counter-current Heat Exchanger Temperature Profile (t={t:.3f}s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_profile.png', dpi=150)
    print("Saved: temperature_profile.png")

    T_outlet_facets = locate_entities_boundary(domain, fdim,
        lambda x: np.isclose(x[0], x0+L, atol=1e-6) & (np.sqrt((x[1]-y0)**2+(x[2]-z0)**2) < r1))
    T_outlet_dofs = locate_dofs_topological(V_T, fdim, T_outlet_facets)

    history_t = np.array(history_t)
    history_X_CO2 = np.array(history_X_CO2)
    history_T_outlet = np.array(history_T_outlet)
    history_rate = np.array(history_rate)

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot(history_t, history_X_CO2, 'g-', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('CO2 Conversion')
    axes[0].set_title('Conversion vs Time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_t, history_T_outlet, 'r-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Outlet Temperature (K)')
    axes[1].set_title('Outlet Temperature vs Time')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('conversion_temperature_vs_time.png', dpi=150)

    plt.figure(figsize=(7,5))
    plt.plot(history_X_CO2, history_T_outlet, 'b.-')
    plt.xlabel('CO2 Conversion')
    plt.ylabel('Outlet Temperature (K)')
    plt.title('Conversion vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('conversion_vs_temperature.png', dpi=150)

    valid = history_rate > 1e-15
    plt.figure(figsize=(7,5))
    plt.plot(history_X_CO2[valid], 1.0/history_rate[valid], 'k.-')
    plt.xlabel('CO2 Conversion (X)')
    plt.ylabel('1/(-r_CO2)  [m^3*s/mol]')
    plt.title('Levenspiel Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('levenspiel_plot.png', dpi=150)

    print("All plots saved")




end=time.time()
print(f'running time{(end-start)/3600}hr')