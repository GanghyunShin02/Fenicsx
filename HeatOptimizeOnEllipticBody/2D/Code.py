%%fenicsx -np 4

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

print('a')
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

  gmsh.option.setNumber("Mesh.MeshSizeMax",0.1)
  gmsh.option.setNumber("Mesh.MeshSizeMin",0.05)
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
print(f'd')
shell_tags=cell_tags.find(2)
body_tags=cell_tags.find(1)

alp.x.array[shell_tags]=1e-7
alp.x.array[body_tags]=1.2e-5
j=Constant(domain,dolfinx.default_scalar_type(1.96e3))

#dx=Measure("dx",domain=domain,subdomain_data=cell_tags)


u=TrialFunction(Q)
v=TestFunction(Q)
un=Function(Q)
un.name='temperature'


# 초기온도 300K
un.x.array[:]=300.0
q=Function(Q)
q.x.array[:]=100


dt=1


a=v*u*dx+dt*alp*dot(grad(v),grad(u))*dx
L=v*un*dx+dt*(q/j)*v*ds(3)


from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

uh = Function(Q)
uh.name='temperature'

problem = LinearProblem(a, L, u=uh,petsc_options_prefix='heat',
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

t = 0.0
t_end = 3600*3

tt=[]
uu=[]

with XDMFFile(MPI.COMM_WORLD, "temperature3.xdmf", "w") as xdmf:

    xdmf.write_mesh(domain)
    xdmf.write_function(un, t)

    while t < t_end:

        uh = problem.solve()
        #print(f'*',end='')

        tt.append(t)
        uu.append(np.min(uh.x.array))
        t += dt




        xdmf.write_function(uh, t)

        un.x.array[:] = uh.x.array

if MPI.COMM_WORLD.rank==0:
  tt=np.array(tt)
  uu=np.array(uu)
  print(tt)
  print(uu)
  plt.figure()
  plt.plot(tt,uu)
  plt.savefig('plot.png')
print('s')
