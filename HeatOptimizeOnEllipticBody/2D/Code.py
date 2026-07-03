%%fenicsx -np 2
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
import dolfinx.fem
import ufl
from ufl import (TrialFunction,
                 TestFunction,
                 dx,
                 dot,
                 grad)
import gmsh
import dolfinx.io.gmsh as gmshio


gmsh.initialize()
gdim=2

if MPI.COMM_WORLD.rank==0:
  ellipse=gmsh.model.occ.addDisk(2,1,0,2,1)
  outer=gmsh.model.occ.addDisk(2,1,0,2.1,1.1)
  shell,_=gmsh.model.occ.cut([2,outer],[2,ellipse])
  gmsh.model.occ.synchronize()
  shell_id=shell[0][1]
  gmsh.model.addPhysicalGroup(2,[ellipse],1,'ellipse')
  gmsh.model.addPhysicalGroup(2,[shell_id],2,'shell')
  gmsh.model.mesh.generate(2)

  gmsh.options.setNumber("Mesh.MeshsizeMax",0.01)
  gmsh.options.setNumber("Mesh.MeshsizeMin",0.005)

mesh_data=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim)
domain=mesh_data.mesh

gmsh.finalize()

from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)


