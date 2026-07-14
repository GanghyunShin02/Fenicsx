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

U_outlet = 0.1
u_outlet_val = np.array([U_outlet, 0.0, 0.0], dtype=default_scalar_type)
bc_outlet = dirichletbc(u_outlet_val, outlet_dofs, Vu_out)

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
T_init_outpipe_fluid = 1000.0
T_init_solid = 300.0

T_n = Function(V_T)


Q_T_in  = functionspace(inflow_submesh,  ("Lagrange", 1))
Q_T_out = functionspace(outflow_submesh, ("Lagrange", 1))

T_in_local  = Function(Q_T_in)
T_out_local = Function(Q_T_out)

T_in_local.x.array[:]  = T_init_inpipe_fluid
T_out_local.x.array[:] = T_init_outpipe_fluid
T_in_local.x.scatter_forward()
T_out_local.x.scatter_forward()



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
T_n.x.array[outflow_dofs] = T_init_outpipe_fluid
T_n.x.array[outpipe_dofs] = T_init_solid

T_n.x.scatter_forward()

'''rhoAir0.352877243550102
visAir4.327984201577027e-05
cpAir1140.9999893726915
kAir0.0676771187675638
rhoH20.024561837963520933
visH22.0725536710106445e-05
cpH214991.855835683164
kH20.46038730633793423
'''
#공기
RHO=Constant(outflow_submesh,default_scalar_type(0.352877243550102))
MU=Constant(outflow_submesh,default_scalar_type(4.327984201577027e-05))
cpAir=Constant(outflow_submesh,default_scalar_type(1140.9999893726915))
kAir=Constant(outflow_submesh,default_scalar_type(0.0676771187675638))

#수소
rho=Constant(inflow_submesh,default_scalar_type(0.024561837963520933))
mu=Constant(inflow_submesh,default_scalar_type(2.0725536710106445e-05))
cpH2=Constant(inflow_submesh,default_scalar_type(114991.855835683164))
kH2=Constant(inflow_submesh,default_scalar_type(0.46038730633793423))


tstep = 0.001
dt_out = Constant(outflow_submesh, default_scalar_type(tstep))

U  = TrialFunction(Vu_out)
V  = TestFunction(Vu_out)
P  = TrialFunction(VP_out)
Q  = TestFunction(VP_out)
Un  = Function(Vu_out)
Un1 = Function(Vu_out)
Us  = Function(Vu_out)


P_   = Function(VP_out)
PHI  = Function(VP_out)

def epsilon(v):
    return sym(grad(v))

     

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

        diff = np.abs(u_new.x.array - u_old.x.array).max()
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


#탄소강

rho_iron =7850.0 
k_iron=41.45 
cp_iron=595.1 

Q_dg = functionspace(domain, ("DG", 0))
rhocp_field = Function(Q_dg)
k_field = Function(Q_dg)

rhocp_field.x.array[inflow_cells]  = float(rho.value) * float(cpH2.value)
rhocp_field.x.array[inpipe_cells]  = rho_iron * cp_iron
rhocp_field.x.array[outflow_cells] = float(RHO.value) * float(cpAir.value)
rhocp_field.x.array[outpipe_cells] = rho_iron * cp_iron

k_field.x.array[inflow_cells]  = float(kH2.value)
k_field.x.array[inpipe_cells]  = k_iron
k_field.x.array[outflow_cells] = float(kAir.value)
k_field.x.array[outpipe_cells] = k_iron

rhocp_field.x.scatter_forward()
k_field.x.scatter_forward()

from dolfinx.fem import create_interpolation_data

Vu_full = functionspace(domain, ("Lagrange", 2, (geom_dim,)))
u_full = Function(Vu_full)

domain_cells_all = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
interp_data_u_in  = create_interpolation_data(Vu_full, Vu_in, domain_cells_all)
interp_data_u_out = create_interpolation_data(Vu_full, Vu_out, domain_cells_all)

u_full.interpolate_nonmatching(u_result, domain_cells_all, interp_data_u_in)

# ===== [변경 3] T_out_local interpolate 준비 (기존에 없었음) =====
outflow_cells_local = np.arange(outflow_submesh.topology.index_map(tdim).size_local, dtype=np.int32)
interp_data_T_out = create_interpolation_data(Q_T_out, V_T, outflow_cells_local)

T = TrialFunction(V_T)
w = TestFunction(V_T)

dt_T = Constant(domain, default_scalar_type(tstep))
T_theta = 0.5*T + 0.5*T_n

dx_full = ufl.Measure("dx", domain=domain, subdomain_data=cell_marker)

FT = rhocp_field*inner((T-T_n)/dt_T, w)*dx_full
FT += k_field*inner(grad(T_theta), grad(w))*dx_full
FT += rhocp_field*inner(dot(u_full, grad(T_theta)), w)*dx_full(1)
FT += rhocp_field*inner(dot(u_full, grad(T_theta)), w)*dx_full(3)

h = ufl.CellDiameter(domain)
u_mag_raw = ufl.sqrt(ufl.dot(u_full, u_full))
u_mag_safe = ufl.max_value(u_mag_raw, 1e-3)   # tau 계산 전용 하한선

tau_supg = h / (2 * u_mag_safe)

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
bc_T_outflow_inlet = dirichletbc(default_scalar_type(T_init_outpipe_fluid), T_outflow_inlet_dofs, V_T)

bcs_T = [bc_T_inflow_inlet, bc_T_outflow_inlet]



A_inlet = np.pi * r1**2
U_inlett = U_inlet

m_dot = float(rho.value) * A_inlet * U_inlett
M_H2 = 0.002016

n_dot_total = m_dot / M_H2

x_CO2 = 1/5
x_H2  = 4/5

n_dot_CO2 = x_CO2 * n_dot_total
n_dot_H2  = x_H2  * n_dot_total

Q_vol = A_inlet * U_inlett

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
    return rate_constant(T) * C_H2_safe**0.88 * C_CO2_safe**0.34

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

r_expr = reaction_rate(T_in_local, C_H2_n, C_CO2_n)

def species_form(C_trial, C_n, D_i, nu_i, w_C):
    return (
        (C_trial - C_n)/dt_C * w_C * ufl.dx
        + dot(u_result, grad(C_trial)) * w_C * ufl.dx
        + D_i * dot(grad(C_trial), grad(w_C)) * ufl.dx
        - nu_i * r_expr * w_C * ufl.dx
    )

F_CO2 = species_form(C_trial, C_CO2_n, D_CO2, -1, w_C)
F_H2  = species_form(C_trial, C_H2_n,  D_H2,  -4, w_C)
F_CH4 = species_form(C_trial, C_CH4_n, D_CH4, +1, w_C)
F_H2O = species_form(C_trial, C_H2O_n, D_H2O, +2, w_C)

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
interp_data_r_in = create_interpolation_data(V_T, Q_r_in, domain_cells_all)

dHrxn = -165000.0
FT_with_reaction = FT - dHrxn * r_field * w * dx_full(1)

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
F11 +=0.5*MU*inner(grad(V), grad(U+Un))*dx
F11 -= dot(P_, div(V))*dx

a11 = form(lhs(F11))
l11 = form(rhs(F11))
A11 = create_matrix(a11)
L11 = create_vector(extract_function_spaces(l11))

a22 = form(dot(grad(Q), grad(P))*dx)
l22 = form(-(RHO/dt_out)*dot(div(Us), Q)*dx)
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
T_end = 0.05
num_steps = int(T_end / tstep)

print("T_outflow_inlet_dofs count:", len(T_outflow_inlet_dofs))
print("rhocp_field min/max:", rhocp_field.x.array.min(), rhocp_field.x.array.max())
print("k_field min/max:", k_field.x.array.min(), k_field.x.array.max())

T_outlet_facets_main = locate_entities_boundary(domain, fdim,
    lambda x: np.isclose(x[0], x0+L, atol=1e-6) & (np.sqrt((x[1]-y0)**2+(x[2]-z0)**2) < r1))
T_outlet_dofs_main = locate_dofs_topological(V_T, fdim, T_outlet_facets_main)


print("wall facets", len(outflow_wall_facets))
print("wall dofs", len(outflow_wall_dofs))
print("velocity inlet", len(outlet_dofs))
print("pressure outlet", len(outflow_outlet_dofsP))



for step in range(num_steps):
    t += tstep
    if MPI.COMM_WORLD.rank==0: 
        print(f'{100*step/num_steps:.1f}% 완료, t={t:.4f}')

    Un1.x.array[:] = Un.x.array
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

    print('1단계직후')
    print("Us min/max:", Us.x.array.min(), Us.x.array.max())
    print("PHI min/max:", PHI.x.array.min(), PHI.x.array.max())
    # ===== Step2: 압력 보정 (복원 필요) =====
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

    print('2단계직후')
    print("Us min/max:", Us.x.array.min(), Us.x.array.max())
    print("PHI min/max:", PHI.x.array.min(), PHI.x.array.max())

    # ===== Step3: 속도 보정 (복원 필요) =====
    with L33.localForm() as loc:
        loc.set(0)
    assemble_vector(L33, l33)
    L33.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver33.solve(L33, Un.x.petsc_vec)
    Un.x.scatter_forward()

    print('3단계직후')
    print("Us min/max:", Us.x.array.min(), Us.x.array.max())
    print("PHI min/max:", PHI.x.array.min(), PHI.x.array.max())


    u_full.interpolate_nonmatching(Un, domain_cells_all, interp_data_u_out)
    u_full.x.scatter_forward()

    T_out_local.interpolate_nonmatching(T_n, outflow_cells_local, interp_data_T_out)
    T_out_local.x.scatter_forward()


    # ===== 여기에 추가 =====
    if MPI.COMM_WORLD.rank==0:
        min_idx = np.argmin(T_out_local.x.array)
        coords = Q_T_out.tabulate_dof_coordinates()
        print("최저온도 위치:", coords[min_idx])
        print("최저온도 값:", T_out_local.x.array[min_idx])
    # ===== 여기까지 =====
    # ===== 조성방정식 (새로 추가) =====
    for A, a, l, bc, solver, LL, C_target in [
        (A_CO2, a_CO2, l_CO2, bc_C_CO2_inlet, solver_CO2, L_CO2, C_CO2_new),
        (A_H2,  a_H2,  l_H2,  bc_C_H2_inlet,  solver_H2,  L_H2,  C_H2_new),
        (A_CH4, a_CH4, l_CH4, bc_C_CH4_inlet, solver_CH4, L_CH4, C_CH4_new),
        (A_H2O, a_H2O, l_H2O, bc_C_H2O_inlet, solver_H2O, L_H2O, C_H2O_new),
    ]:
        A.zeroEntries()
        assemble_matrix(A, a, bcs=[bc])
        A.assemble()
        with LL.localForm() as loc:
            loc.set(0)
        assemble_vector(LL, l)
        apply_lifting(LL, [a], [[bc]])
        LL.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(LL, [bc])
        solver.solve(LL, C_target.x.petsc_vec)
        C_target.x.scatter_forward()

    C_CO2_n.x.array[:] = C_CO2_new.x.array
    C_H2_n.x.array[:]  = C_H2_new.x.array
    C_CH4_n.x.array[:] = C_CH4_new.x.array
    C_H2O_n.x.array[:] = C_H2O_new.x.array

    # ===== 반응속도 -> 전체 mesh 반영 (새로 추가) =====
    r_field_local.interpolate(fem.Expression(
        reaction_rate(T_in_local, C_H2_n, C_CO2_n), Q_r_in.element.interpolation_points))
    r_field.interpolate_nonmatching(r_field_local, domain_cells_all, interp_data_r_in)
    r_field.x.scatter_forward()



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

    vtx_u_out.write(t)
    vtx_p_out.write(t)
    vtx_T.write(t)

    C_CO2_outlet_avg = C_CO2_n.x.array[outlet_dofs_C].mean() if len(outlet_dofs_C) > 0 else np.nan
    X_CO2_now = (C_CO2_init - C_CO2_outlet_avg) / C_CO2_init

    T_outlet_avg = T_n.x.array[T_outlet_dofs_main].mean() if len(T_outlet_dofs_main) > 0 else np.nan
    rate_avg = r_field_local.x.array.mean()

    history_t.append(t)
    history_X_CO2.append(X_CO2_now)
    history_T_outlet.append(T_outlet_avg)
    history_rate.append(rate_avg)

    if MPI.COMM_WORLD.rank==0:
        print(f"X_CO2={X_CO2_now}  T_outlet={T_outlet_avg}")
        print("T_out_local min/max:", T_out_local.x.array.min(), T_out_local.x.array.max())
        print("Un min/max:", Un.x.array.min(), Un.x.array.max())
        print("T_n min/max:", T_n.x.array.min(), T_n.x.array.max())
        print("P min/max:", P_.x.array.min(), P_.x.array.max())
        print("||A11|| =", A11.norm())
        print("||L11|| =", L11.norm())
        conv = 1.5*Un.x.array - 0.5*Un1.x.array
        print("conv min/max:", conv.min(), conv.max())
        div_form = fem.form(ufl.div(Un) * ufl.dx)
        div_val = fem.assemble_scalar(div_form)
        print("Integral div(U):", div_val)
        print("NaN(Un):", np.isnan(Un.x.array).any(), " NaN(T_n):", np.isnan(T_n.x.array).any())


vtx_u_in.close()
vtx_p_in.close()
vtx_u_out.close()
vtx_p_out.close()
vtx_T.close()

if MPI.COMM_WORLD.rank==0:
    print("inflow cells:", inflow_submesh.topology.index_map(inflow_submesh.topology.dim).size_global)

    T_dof_coords = V_T.tabulate_dof_coordinates()

    def get_avg_T_at_x_bin(x_target, x_tol, r_min, r_max):

        print(x_target)
        print(type(x_target))
        print(np.shape(x_target))
        
        print(f'T_dof_coords{T_dof_coords.shape}')
        print(f'Tn.x.array{T_n.x.array.shape}')
        x_coords = T_dof_coords[:, 0]
        y_coords = T_dof_coords[:, 1]
        z_coords = T_dof_coords[:, 2]
        r = np.sqrt((y_coords - y0)**2 + (z_coords - z0)**2)

        mask = (np.abs(x_coords - x_target) < x_tol) & (r >= r_min) & (r <= r_max)
        if mask.sum() == 0:
            return np.nan
        return T_n.x.array[mask].mean()

    n_points = 50
    print(f'L,{L}')

    x_samples = np.linspace(x0, x0+L, n_points)
    x_tol = L / n_points

    T_inflow_profile  = [get_avg_T_at_x_bin(j, x_tol, 0, r1*0.9) for j in x_samples]
    T_outflow_profile = [get_avg_T_at_x_bin(k, x_tol, r2*1.05, r3*0.95) for k in x_samples]

    plt.figure(figsize=(8,5))
    plt.plot(x_samples, T_inflow_profile, 'r-', linewidth=2, label='Inner tube fluid (H2)')
    plt.plot(x_samples, T_outflow_profile, 'b-', linewidth=2, label='Outer tube fluid (Air)')
    plt.xlabel('Tube length (m)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Counter-current Heat Exchanger Temperature Profile (t={t:.3f}s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig('11temperature_profile.png', dpi=150)

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
    plt.show()
    plt.savefig('11conversion_temperature_vs_time.png', dpi=150)

    plt.figure(figsize=(7,5))
    plt.plot(history_X_CO2, history_T_outlet, 'b.-')
    plt.xlabel('CO2 Conversion')
    plt.ylabel('Outlet Temperature (K)')
    plt.title('Conversion vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig('conversion_vs_temperature.png', dpi=150)

    valid = history_rate > 1e-15
    plt.figure(figsize=(7,5))
    plt.plot(history_X_CO2[valid], 1.0/history_rate[valid], 'k.-')
    plt.xlabel('CO2 Conversion (X)')
    plt.ylabel('1/(-r_CO2)  [m^3*s/mol]')
    plt.title('Levenspiel Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig('11levenspiel_plot.png', dpi=150)
    

    print("All plots saved")




end=time.time()
print(f'running time{(end-start)/3600}hr')