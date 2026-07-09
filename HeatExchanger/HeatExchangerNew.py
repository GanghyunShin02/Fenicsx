import numpy as np
import matplotlib.pyplot as plt

import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx import fem
from dolfinx import mesh,io
from dolfinx.io import VTXWriter,gmsh as gmshio
from dolfinx.mesh import (locate_entities_boundary,
                            create_submesh)
import ufl
from ufl import (grad,
                 dot,
                 inner,
                 TrialFunction,
                 TestFunction)
from dolfinx.fem import (Function,
                         functionspace,
                         dirichletbc,
                         locate_dofs_topological)

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

u_nonslip=np.array((0,0,0),dtype=dolfinx.default_scalar_type)


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

U_inlet = 1  # 유입 속도 크기 (예시)
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




