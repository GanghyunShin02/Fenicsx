
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import dolfinx
import dolfinx.fem 
from dolfinx.fem import (Constant,
                         functionspace,
                         Function)
import ufl
from ufl import Measure
from ufl import (TrialFunction,
                 TestFunction,
                 dx,
                 dot,
                 grad,
               )
import gmsh
import dolfinx.io.gmsh as gmshio


gmsh.initialize()
gdim=2

if MPI.COMM_WORLD.rank==0:
  ellipse1=gmsh.model.occ.addDisk(2,1,0,2,1)
  outer=gmsh.model.occ.addDisk(2,1,0,2.1,1.1)
  obj,_=gmsh.model.occ.fragment([(2,outer)],[(2,ellipse1)])
  gmsh.model.occ.synchronize()
  shell=obj[0][1]
  body=obj[1][1]

 


  gmsh.model.addPhysicalGroup(2,[body],1,'body')
  gmsh.model.addPhysicalGroup(2,[shell],2,'shell')

  gmsh.option.setNumber("Mesh.MeshSizeMax",0.01)
  gmsh.option.setNumber("Mesh.MeshSizeMin",0.005)
  gmsh.model.mesh.generate(2)



mesh_data=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim)
domain=mesh_data.mesh

cell_tags=mesh_data.cell_tags
facet_tags=mesh_data.facet_tags

#위껍질 locate(..)는 인덱스를 반환..태그를 내가 붙여야 써먹을수있다.
topshell=dolfinx.mesh.locate_entities_boundary(domain,1,
                                  lambda x:x[1]>1.1)

topshell_tag=dolfinx.mesh.meshtags(domain,
                                   1,
                                   topshell,
                                   np.full(len(topshell),3,dtype=np.int32))

ds=Measure("ds",domain=domain,subdomain_data=topshell_tag)

gmsh.finalize()

'''
from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "2Dmesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    '''

Q=functionspace(domain,('CG',1))
QQ=functionspace(domain,('DG',0))
alp=Function(QQ)

shell_tags=cell_tags.find(2)
body_tags=cell_tags.find(1)

alp.x.array[shell_tags]=100
alp.x.array[body_tags]=10

#dx=Measure("dx",domain=domain,subdomain_data=cell_tags)


u=TrialFunction(Q)
v=TestFunction(Q)
un=Function(Q)


# 초기온도 300K
un.x.array[:]=300.0
q=Function(Q)
q.x.array[:]=100

T=100
dt=1


a=v*u*dx+dt*alp*dot(grad(v),grad(u))*dx
L=v*un*dx+dt*q*v*ds(3)


from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

uh = Function(Q)

problem = LinearProblem(a, L, u=uh,petsc_options_prefix='heat')

t = 0.0
t_end = 100.0

with XDMFFile(MPI.COMM_WORLD, "temperature.xdmf", "w") as xdmf:

    xdmf.write_mesh(domain)
    xdmf.write_function(un, t)

    while t < t_end:

        uh = problem.solve()
        print(f'진행중..{t/t_end}%')
        t += dt

  

        xdmf.write_function(uh, t)

        un.x.array[:] = uh.x.array
