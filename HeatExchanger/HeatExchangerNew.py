import numpy as np
import matplotlib.pyplot as plt

import dolfinx
from mpi4py import MPI
import gmsh
from dolfinx import fem
from dolfinx import mesh,io
from dolfinx.io import VTXWriter,gmsh as gmshio
import ufl
from ufl import (grad,
                 dot,
                 inner,
                 TrialFunction,
                 TestFunction)
from dolfinx.fem import Function

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

with io.XDMFFile(domain.comm, "heat_exchanger3D_geometry.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
    xdmf.write_meshtags(cell_marker, domain.geometry)