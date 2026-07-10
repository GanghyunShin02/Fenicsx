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

gmsh.initialize()

if MPI.COMM_WORLD.rank == 0:
    r1, r2, r3, r4 = 1, 1.1, 2.1, 2.2
    x0, y0, z0, L = 0, 0, 2.2, 10

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
    # inpipe(내관 solid)의 안쪽면/바깥면 찾기
    inpipe_boundary = gmsh.model.getBoundary(
        [(3, t) for t in inpipe], oriented=False, combined=False
    )
    # outpipe(외관 solid)의 안쪽면/바깥면 찾기
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

'''
with io.XDMFFile(domain.comm, "heat_exchanger3D_geometry.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
    xdmf.write_meshtags(cell_marker, domain.geometry)
'''

# 경계조건


tdim = domain.topology.dim
fdim = tdim - 1

# cell_marker 태그: 1=inflow, 2=inpipe, 3=outflow, 4=outpipe
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

# --- inflow: 벽면은 r=r1 (inpipe 안쪽면과 접촉) ---
def inflow_wall(x):
    r = np.sqrt((x[1]-y0)**2 + (x[2]-z0)**2)
    return np.isclose(r, r1, atol=1e-3)

inflow_wall_facets = locate_entities_boundary(inflow_submesh, fdim, inflow_wall)
inflow_wall_dofs   = locate_dofs_topological(Vu_in, fdim, inflow_wall_facets)
bc_inflow_wall = dirichletbc(u_zero, inflow_wall_dofs, Vu_in)

# --- outflow: 벽면은 r=r2 (outpipe 안쪽) 와 r=r3 (outpipe 바깥쪽) ---
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

# inflow 유입부: x = x0 (=0)
def inlet_face(x):
    return np.isclose(x[0], x0, atol=1e-6)

inlet_facets = locate_entities_boundary(inflow_submesh, fdim, inlet_face)
inlet_dofs   = locate_dofs_topological(Vu_in, fdim, inlet_facets)

U_inlet = 1  # 유입 속도 크기 
u_inlet_val = np.array([U_inlet, 0.0, 0.0], dtype=default_scalar_type)
bc_inlet = dirichletbc(u_inlet_val, inlet_dofs, Vu_in)

# outflow 출구부: x = x0 + L, 압력 p=0 (Neumann은 자연 경계조건이라 별도 BC 불필요, p만 예시)

def outlet_face(x):
    return np.isclose(x[0], x0 + L, atol=1e-6)
# 병ㄹ면 return np.isclose(x[0],x0,atol=1e-6)

outlet_facets = locate_entities_boundary(outflow_submesh, fdim, outlet_face)
outlet_dofs   = locate_dofs_topological(Vu_out, fdim, outlet_facets)

U_outlet=10
u_outlet_val=np.array([U_outlet,0.0,0.0],dtype=default_scalar_type)
bc_outlet=dirichletbc(u_outlet_val,outlet_dofs,Vu_out)

# 압력경계조건
def outlet_faceP(x):
    return np.isclose(x[0], x0 + L, atol=1e-6)  # inflow의 경우

inflow_outlet_facetsP = locate_entities_boundary(inflow_submesh, fdim, outlet_face)
inflow_outlet_dofsP = locate_dofs_topological(VP_in, fdim, inflow_outlet_facetsP)

p_outlet_val = default_scalar_type(0.0)
bc_outlet_p = dirichletbc(p_outlet_val, inflow_outlet_dofsP, VP_in)

def outflow_outlet_faceP(x):
    return np.isclose(x[0], x0 , atol=1e-6)  # outflow의 경우

outflow_outlet_facetsP = locate_entities_boundary(outflow_submesh, fdim, outflow_outlet_faceP)
outflow_outlet_dofsP = locate_dofs_topological(VP_out, fdim, outflow_outlet_facetsP)    

P_outflow_outlet_val = default_scalar_type(0.0)
bc_outflow_outlet_P = dirichletbc(P_outflow_outlet_val, outflow_outlet_dofsP, VP_out)



# 초기온도

V_T=functionspace(domain,("Lagrange",1))

T_init_inpipe_fluid  = 500.0  # 내관유체 (수증기)
T_init_outpipe_fluid = 300.0  # 외관유체
T_init_solid = 300.0          # 관(내관+외관 solid) 전체

T_n = Function(V_T)   # 전체 domain 온도장

# cell_marker: 1=inflow(내관유체), 2=inpipe(내관벽), 3=outflow(외관유체), 4=outpipe(외관벽)

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

T_n.x.scatter_forward()  # MPI 병렬 환경 필수


# 상수
R_specific = 461.5     # J/(kg·K), 수증기 비기체상수
P_steam = 101325.0     # Pa, 운전 압력 (실제 값으로 수정)

mu_ref_steam = 1.12e-5  # Pa·s
T_ref_steam  = 350.0    # K
S_steam      = 1064.0   # Sutherland 상수 (수증기)

def rho_steam(T):
    return P_steam / (R_specific * T)

def mu_steam(T):
    return mu_ref_steam * (T/T_ref_steam)**1.5 * (T_ref_steam + S_steam) / (T + S_steam)

def rho_water(T):
    return 1000.0 - 0.0178 * (T - 277.0)**1.7   # kg/m^3

def mu_water(T):
    A, B, C = 2.414e-5, 247.8, 140.0
    return A * 10**(B / (T - C))                 # Pa·s


# 내관은 u,외관은 U
tstep=0.1
dt_in=Constant(inflow_submesh,default_scalar_type(tstep))
dt_out=Constant(outflow_submesh,default_scalar_type(tstep))

u=TrialFunction(Vu_in)
v=TestFunction(Vu_in)
p=TrialFunction(VP_in)
q=TestFunction(VP_in)
un=Function(Vu_in)
un1=Function(Vu_in)
us=Function(Vu_in)
rho=Constant(inflow_submesh,default_scalar_type(float(rho_water(300))))
mu=Constant(inflow_submesh,default_scalar_type(float(mu_water(300))))
p_=Function(VP_in)
phi=Function(VP_in)

U=TrialFunction(Vu_out)
V=TestFunction(Vu_out)
P=TrialFunction(VP_out)
Q=TestFunction(VP_out)
Un=Function(Vu_out)
Un1=Function(Vu_out)
Us=Function(Vu_out)
RHO=Constant(outflow_submesh,default_scalar_type(float(rho_steam(500))))
MU=Constant(outflow_submesh,default_scalar_type(float(mu_steam(500))))
P_=Function(VP_out)
PHI=Function(VP_out)


#속도추정

def epsilon(v): #응력텐서
    return sym(grad(v))



print("rho type:", type(rho), "rho shape:", rho.ufl_shape if hasattr(rho, 'ufl_shape') else "NO SHAPE ATTR")
print("dt_in type:", type(dt_in))


F1=(rho/dt_in)*dot((u-un),v)*dx
F1+=rho*inner(
    dot(
        (1.5*un-0.5*un1),
        (0.5*nabla_grad(u+un))
    ),
    v
)*dx
F1+=2*0.5*mu*inner(epsilon(v),
               epsilon(u+un))*dx
F1-=dot(p_,div(v))*dx

a1=form(lhs(F1))
l1=form(rhs(F1))
A1=create_matrix(a1)
L1=create_vector(extract_function_spaces(l1))

a2=form(dot(grad(q),grad(p))*dx)
l2=form(-(rho/dt_in)*dot(div(us),q)*dx)
A2=assemble_matrix(a2,bcs=[bc_outlet_p])
A2.assemble()
L2=create_vector(extract_function_spaces(l2))

a3=form(rho*dot(u,v)*dx)
l3=form(rho*dot(us,v)*dx-dt_in*dot(nabla_grad(phi),v)*dx)
A3=assemble_matrix(a3)
A3.assemble()
L3=create_vector(extract_function_spaces(l3))



F11=(RHO/dt_out)*dot((U-Un),V)*dx
F11+=RHO*inner(
    dot(
        (1.5*Un-0.5*Un1),
        (0.5*nabla_grad(U+Un))
    ),
    V
)*dx
F11+=2*0.5*MU*inner(epsilon(V),
                  epsilon(U+Un))*dx
F11-=dot(P_,div(V))*dx

a11=form(lhs(F11))
l11=form(rhs(F11))
A11=create_matrix(a11)
L11=create_vector(extract_function_spaces(l11))

a22=form(dot(grad(Q),grad(P))*dx)
l22=form(-(RHO/dt_out)*dot(div(Us),Q)*dx)
A22=assemble_matrix(a22,bcs=[bc_outflow_outlet_P])
A22.assemble()
L22=create_vector(extract_function_spaces(l22))

a33=form(RHO*dot(U,V)*dx)
l33=form(RHO*dot(Us,V)*dx-dt_out*dot(nabla_grad(PHI),V)*dx)
A33=assemble_matrix(a33)
A33.assemble()
L33=create_vector(extract_function_spaces(l33))


# Solver for step 1
solver1 = PETSc.KSP().create(inflow_submesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.GMRES)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(inflow_submesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(inflow_submesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# Solver for step 1
solver11 = PETSc.KSP().create(outflow_submesh.comm)
solver11.setOperators(A11)
solver11.setType(PETSc.KSP.Type.GMRES)
pc11 = solver11.getPC()
pc11.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver22 = PETSc.KSP().create(outflow_submesh.comm)
solver22.setOperators(A22)
solver22.setType(PETSc.KSP.Type.MINRES)
pc22 = solver22.getPC()
pc22.setType(PETSc.PC.Type.HYPRE)
pc22.setHYPREType("boomeramg")

# Solver for step 3
solver33 = PETSc.KSP().create(outflow_submesh.comm)
solver33.setOperators(A33)
solver33.setType(PETSc.KSP.Type.CG)
pc33 = solver33.getPC()
pc33.setType(PETSc.PC.Type.SOR)

# --- VTX 출력 준비 ---
vtx_u_in  = VTXWriter(inflow_submesh.comm,  "results/4u_inflow1.bp",  [un],  engine="BP4")
vtx_p_in  = VTXWriter(inflow_submesh.comm,  "results/4p_inflow1.bp",  [p_],  engine="BP4")
vtx_u_out = VTXWriter(outflow_submesh.comm, "results/4u_outflow1.bp", [Un],  engine="BP4")
vtx_p_out = VTXWriter(outflow_submesh.comm, "results/4p_outflow1.bp", [P_],  engine="BP4")

# --- 시간 루프 ---
t = 0.0
T_end = 1
num_steps = int(T_end / tstep)

print("dt_in value:", dt_in.value)
print("dt_out value:", dt_out.value)


for step in range(num_steps):
    t += tstep
    print(f'{100*step/num_steps:.1f}% 완료, t={t:.3f}')

    # ===== inflow =====
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=[bc_inflow_wall, bc_inlet])
    A1.assemble()

    A11.zeroEntries()
    assemble_matrix(A11, a11, bcs=[bc_outflow_wall, bc_outlet])
    A11.assemble()

    # Step 1: 속도 추정 (us)
    with L1.localForm() as loc:
        loc.set(0)
    assemble_vector(L1, l1)
    apply_lifting(L1, [a1], [[bc_inflow_wall, bc_inlet]])
    L1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L1, [bc_inflow_wall, bc_inlet])
    solver1.solve(L1, us.x.petsc_vec)
    us.x.scatter_forward()

    # Step 2: 압력 보정 (phi)
    with L2.localForm() as loc:
        loc.set(0)
    assemble_vector(L2, l2)
    apply_lifting(L2, [a2], [[bc_outlet_p]])
    L2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L2, [bc_outlet_p])
    solver2.solve(L2, phi.x.petsc_vec)
    phi.x.scatter_forward()

    p_.x.array[:] += phi.x.array
    p_.x.scatter_forward()

    # Step 3: 속도 보정
    with L3.localForm() as loc:
        loc.set(0)
    assemble_vector(L3, l3)
    L3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(L3, un.x.petsc_vec)
    un.x.scatter_forward()

    un1.x.array[:] = un.x.array

    # ===== outflow =====
    with L11.localForm() as loc:
        loc.set(0)
    assemble_vector(L11, l11)
    apply_lifting(L11, [a11], [[bc_outflow_wall, bc_outlet]])
    L11.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L11, [bc_outflow_wall, bc_outlet])
    solver11.solve(L11, Us.x.petsc_vec)
    Us.x.scatter_forward()

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

    with L33.localForm() as loc:
        loc.set(0)
    assemble_vector(L33, l33)
    L33.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver33.solve(L33, Un.x.petsc_vec)
    Un.x.scatter_forward()

    Un1.x.array[:] = Un.x.array

    # ===== 저장 =====
    vtx_u_in.write(t)
    vtx_p_in.write(t)
    vtx_u_out.write(t)
    vtx_p_out.write(t)

    print("un array min/max:", un.x.array.min(), un.x.array.max())
    print("Un array min/max:", Un.x.array.min(), Un.x.array.max())
    print("NaN 있음?:", np.isnan(un.x.array).any())
    print("NaN 있음?:", np.isnan(Un.x.array).any())

vtx_u_in.close()
vtx_p_in.close()
vtx_u_out.close()
vtx_p_out.close()

print("inflow cells:", inflow_submesh.topology.index_map(inflow_submesh.topology.dim).size_global)
