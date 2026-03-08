# Heat equation.

First, this is simulating Heat equation.

$$
u_t=\nabla^2u+f $$

$$
 u=u_0=exp(-a(x^2+y^2))\\ at\\ t=0 $$

$$
\\ u=u_D \\ at \\ \partial \Omega
$$

To compute this PDE, we need to time-variative.

$$
(\frac{\partial u}{\partial t})^{n+1}=\frac{u^{n+1}-u^n}{\Delta t} $$

like this. Then,

$$
\\ \frac{u^{n+1}-u^n}{\Delta t}=\nabla^2 u^{n+1}+f^{n+1}$$
$$
\\ u^{n+1}=u^n+\Delta t \nabla^2u^{n+1}+\Delta t f^{n+1}
$$

Change Week fomulation. (v is 0 at boundary conditon.)

$$
\int_{\Omega}vu^{n+1}dx-\int_{\Omega}\Delta t \; v\nabla^2u^{n+1}dx=\int_{\Omega}v(u^n+\Delta t f^{n+1})dx$$
$$
\\ \int_{\Omega}(vu^{n+1}+\Delta t \nabla v \; \nabla u^{n+1})dx=\int_{\Omega}v(u^n+\Delta t f^{n+1})$$
$$
a(u,v)=\int_{\Omega}(vu^{n+1}+\Delta t \nabla v \; \nabla u^{n+1})dx$$
$$
L(v)=\int_{\Omega}v(u^n+\Delta t f^{n+1})$$
