"""Fri Sep 30 10:39:39 CDT 2022"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import pyopencl.tools as cl_tools
from functools import partial
from dataclasses import dataclass, fields

#from arraycontext import thaw, freeze
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.dof_desc import BoundaryDomainTag
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, VolumeDomainTag
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    AdiabaticSlipBoundary,
)

from mirgecom.fluid import (
    velocity_gradient, make_conserved
)

from mirgecom.transport import SimpleTransport

from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)

from pytools.obj_array import make_obj_array

from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_ns_heat_operator
)

from mirgecom.diffusion import (
    diffusion_operator, DirichletDiffusionBoundary, NeumannDiffusionBoundary
)

#########################################################################

class Initializer:

    def __init__(self, *, dim=2):

        self._dim = dim

    def __call__(self, x_vec, eos):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        actx = x_vec[0].array_context

        mom_x = 0.0*x_vec[0]
        mom_y = 0.0*x_vec[0]
        momentum = make_obj_array([mom_x, mom_y])

        temperature = 300.0 + 0.0*x_vec[0]
        mass = 1.0 + 0.0*x_vec[0]

        #~~~ 
        energy = eos.get_internal_energy(temperature)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                momentum=momentum)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


#h1 = logging.StreamHandler(sys.stdout)
#f1 = SingleLevelFilter(logging.INFO, False)
#h1.addFilter(f1)
#root_logger = logging.getLogger()
#root_logger.addHandler(h1)
#h2 = logging.StreamHandler(sys.stderr)
#f2 = SingleLevelFilter(logging.INFO, True)
#h2.addFilter(f2)
#root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(
            self,
            heat_capacity,
            thermal_conductivity_func,
            *,
            effective_surface_area_func=None,
            mass_loss_func=None,
            oxygen_diffusivity=0.):
        self._heat_capacity = heat_capacity
        self._thermal_conductivity_func = thermal_conductivity_func
        self._effective_surface_area_func = effective_surface_area_func
        self._mass_loss_func = mass_loss_func
        self._oxygen_diffusivity = oxygen_diffusivity

    @property
    def heat_capacity(self):
        return self._heat_capacity

    def thermal_conductivity(self, mass, temperature):
        return self._thermal_conductivity_func(mass, temperature)

    def thermal_diffusivity(self, mass, temperature, thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(mass, temperature)
        return thermal_conductivity/(mass * self.heat_capacity)

    def mass_loss_rate(self, mass, ox_mass, temperature):
        dm = mass*0.
        if self._effective_surface_area_func is not None:
            eff_surf_area = self._effective_surface_area_func(mass)
            if self._mass_loss_func is not None:
                dm = self._mass_loss_func(mass, ox_mass, temperature, eff_surf_area)
        return dm

    @property
    def oxygen_diffusivity(self):
        return self._oxygen_diffusivity

    def temperature(self, wv):
        return wv.energy/(wv.mass * self.heat_capacity)

    def dependent_vars(self, wv):
        temperature = self.temperature(wv)
        kappa = self.thermal_conductivity(wv.mass, temperature)
        return WallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # ~~~~~~~~~~~~~~~~~~

    mesh_filename = "grid-v2.msh"

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 100
    nrestart = 20000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 1.0e-10
    t_final = 2.0

    constant_cfl = False
    current_cfl = 0.2
    
    # discretization and model control
    order = 1

    use_radiation = False

##########################################################################
    
    current_step = 0
    current_t = 0.0
    dim = 2

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")


##########################################################################

    nspecies = None

    eos = IdealSingleGas()

    # }}}    
    
    # {{{ Initialize transport model
    physical_transport = SimpleTransport(viscosity=0.0,
        thermal_conductivity=0.0)

    gas_model = GasModel(eos=eos, transport=physical_transport)

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # for graphite
    wall_insert_rho = 1625
    wall_insert_cp = 770
    wall_insert_kappa = 247.5

    # wall stuff
    wall_penalty_amount = 0.0 #25
    wall_time_scale = 1

    temp_wall = 1500

#############################################################################

    flow_init = Initializer(dim=dim)

##############################################################################

    restart_step = None
    if restart_file is None:        
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "fluid": ["fluid"],
                "solid": ["wall_insert"]
                }
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    local_nelements = (
          volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["solid"][0].nelements)

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order)


    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)

    wall_vol_discr = dcoll.discr_from_dd(dd_vol_solid)
    wall_tag_to_elements = volume_to_local_mesh_data["solid"][1]
    wall_insert_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_insert"])

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))

    #~~~~~~~~~~

    from grudge.dt_utils import characteristic_lengthscales
    char_length_fluid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_solid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid)

#####################################################################################

    def _create_wall_dependent_vars(wv):
        return wall_model.dependent_vars(wv)

    create_wall_dependent_vars_compiled = actx.compile(
        _create_wall_dependent_vars)

    def _get_wv(wv):
        return wv

    get_wv = actx.compile(_get_wv)

    #~~~~~~~~~~~~~~~~~~

    def _get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)

    get_fluid_state = actx.compile(_get_fluid_state)

    tseed = None

#####################################################################################

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x,
                                       initial=np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x,
                                       initial=-np.inf))[()]

#########################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.5e} s, "),
            ("t_sim.max", "sim time: {value:1.5e} s, "),
            ("t_step.max", "--- step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max",
                                "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)
        
#################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(fluid_nodes, eos)

        wall_mass = (
            wall_insert_rho * wall_insert_mask
        )
        wall_cp = (
            wall_insert_cp * wall_insert_mask
        )
        current_wv = WallVars(
            mass=wall_mass,
            energy=wall_mass * wall_cp * temp_wall,
            ox_mass=0*wall_mass)

    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t))

        if restart_iterations:
            current_t = 0.0
            current_step = 0

        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol"))

            current_cv = connection(restart_data["state"])
            tseed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["state"]
            tseed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)


    current_cv = force_evaluation(actx, current_cv)
    current_wv = force_evaluation(actx, current_wv)

    current_state = get_fluid_state(current_cv)

#####################################################################################
    
    fluid_boundaries = {
        dd_vol_fluid.trace("sym").domain_tag: AdiabaticSlipBoundary(),
    }

    wall_symmetry = NeumannDiffusionBoundary(0.0)
    solid_boundaries = {
        dd_vol_solid.trace("wall_sym").domain_tag: wall_symmetry
    }

##############################################################################

    def _get_wall_kappa_inert(mass, temperature):
        return wall_insert_kappa * wall_insert_mask

    wall_model = WallModel(
        heat_capacity=wall_insert_cp * wall_insert_mask,
        thermal_conductivity_func=_get_wall_kappa_inert
    )

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     t_initial=current_t,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

    def my_write_viz(step, t, dt, fluid_state, wv, wall_kappa, wall_temperature,
                     fluid_rhs=None, wall_energy_rhs=None, grad_cv_fluid=None,
                     grad_t_fluid=None, grad_t_solid=None):

        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature),
            ("DV_U", fluid_state.velocity[0]),
            ("DV_V", fluid_state.velocity[1]),
            ("RHS", fluid_rhs),
            ("grad_rho", grad_cv_fluid.mass),
            ("grad_rhoU", grad_cv_fluid.momentum[0]),
            ("grad_rhoV", grad_cv_fluid.momentum[1]),
            ("grad_rhoE", grad_cv_fluid.energy),
            ("grad_T", grad_t_fluid),
            ("mu", fluid_state.tv.viscosity),
            ("kappa", fluid_state.tv.thermal_conductivity),
        ]

        cell_alpha = wall_model.thermal_diffusivity(wv.mass, wall_kappa)
        solid_viz_fields = [
            ("wv", wv),
            ("wall_kappa", wall_kappa),
            ("wall_temperature", wall_temperature),
            ("wall_alpha", cell_alpha),
            ("wall_RHS", wall_energy_rhs),
            ("wall_grad_T", grad_t_solid),
        ]
                     
        print('Writing solution file...')
        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(
            dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t,
            overwrite=True, comm=comm)

    from mirgecom.restart import write_restart_file
    def my_write_restart(step, t, state):
        if rank == 0:
            print('Writing restart file...')

        cv, wv = state
        restart_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "nspecies": nspecies,
                "wv": wv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

#########################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    from grudge.op import nodal_min_loc, nodal_max_loc, nodal_min, nodal_max

    def my_get_wall_timestep(dcoll, wv, wdv):
        wall_diffusivity = wall_model.thermal_diffusivity(wv.mass,
            wdv.temperature, wdv.thermal_conductivity)
        return (
            char_length_solid**2/(wall_time_scale * actx.np.maximum(
                    wall_diffusivity, wall_model.oxygen_diffusivity))
        )

    def _my_get_timestep_wall(dcoll, wv, wdv, t, dt, cfl, t_final,
            constant_cfl=False, local_dt=False, wall_dd=dd_vol_solid):

        if not constant_cfl:
            return dt

        actx = wv.mass.array_context
        if local_dt:
            mydt = cfl*my_get_wall_timestep(dcoll, wv, wdv)            
        else:
            if constant_cfl:
                ts_field = cfl*my_get_wall_timestep(dcoll, wv, wdv)
                mydt = actx.to_numpy(
                    nodal_min(dcoll, wall_dd, ts_field, initial=np.inf))[()]

        return mydt

    my_get_timestep_wall = _my_get_timestep_wall

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, wv = state
        cv = force_evaluation(actx, cv)
        wv = force_evaluation(actx, wv)

        fluid_state = get_fluid_state(cv)

        wdv = create_wall_dependent_vars_compiled(wv)

        if constant_cfl:
            dt_fluid = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                t_final, constant_cfl, local_dt=False, fluid_dd=dd_vol_fluid)
            dt_fluid = force_evaluation(actx, dt_fluid)

            dt_wall = my_get_timestep_wall(
                dcoll=dcoll, wv=wv, wdv=wdv, t=t, dt=current_dt,
                cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
                local_dt=False, wall_dd=dd_vol_solid)
            dt_wall = force_evaluation(actx, dt_wall)

            dt = np.minimum(dt_fluid, dt_wall)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            ngarbage = 1000
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            state = make_obj_array([cv, wv])

            file_exists = os.path.exists('write_solution')
            if file_exists:
              os.system('rm write_solution')
              do_viz = True
        
            file_exists = os.path.exists('write_restart')
            if file_exists:
              os.system('rm write_restart')
              do_restart = True

            if do_health:
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step, t, state)

            if do_viz:
                fluid_rhs, wall_energy_rhs, grad_cv_fluid, grad_t_fluid, grad_t_solid = \
                    coupled_ns_heat_operator(
                        dcoll, gas_model, dd_vol_fluid, dd_vol_solid, fluid_boundaries, solid_boundaries,
                        fluid_state, wdv.thermal_conductivity, wdv.temperature, time=t, quadrature_tag=quadrature_tag,
                        return_gradients=True, interface_radiation=use_radiation, wall_epsilon=1.0, interface_noslip=False,
                    )

                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature, fluid_rhs=fluid_rhs,
                    wall_energy_rhs=wall_energy_rhs, grad_cv_fluid=grad_cv_fluid,
                    grad_t_fluid=grad_t_fluid, grad_t_solid=grad_t_solid)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature)
            raise

        return make_obj_array([fluid_state.cv, wv]), dt


    def my_rhs(t, state):
        cv, wv = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
        wdv = wall_model.dependent_vars(wv)

        #~~~~~~~~~~~~~
        fluid_rhs, wall_energy_rhs = coupled_ns_heat_operator(
                dcoll, gas_model, dd_vol_fluid, dd_vol_solid, fluid_boundaries,
                solid_boundaries, fluid_state, wdv.thermal_conductivity,
                wdv.temperature, time=t, quadrature_tag=quadrature_tag,
                interface_radiation=use_radiation, wall_epsilon=1.0, interface_noslip=False,
            )

        #~~~~~~~~~~~~~
        wall_mass_rhs = -wall_model.mass_loss_rate(mass=wv.mass,
            ox_mass=wv.ox_mass, temperature=wdv.temperature)

        wall_energy_rhs = wall_time_scale * wall_energy_rhs

        wall_ox_mass_rhs = wall_mass_rhs * 0.0

        wall_rhs = WallVars(mass=wall_mass_rhs, energy=wall_energy_rhs,
            ox_mass=wall_ox_mass_rhs)

        return make_obj_array([fluid_rhs, wall_rhs])

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)         
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([current_state.cv, current_wv])

    if constant_cfl:
        dt_fluid = get_sim_timestep(dcoll, current_state, current_t, current_dt, current_cfl,
                          t_final, constant_cfl, local_dt=False, fluid_dd=dd_vol_fluid)
        dt_fluid = force_evaluation(actx, dt_fluid)

        wdv = wall_model.dependent_vars(current_wv)
        dt_wall = my_get_timestep_wall(
            dcoll=dcoll, wv=current_wv, wdv=wdv, t=current_t, dt=current_dt,
            cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
            local_dt=False, wall_dd=dd_vol_solid)
        dt_wall = force_evaluation(actx, dt_wall)

        dt = np.minimum(dt_fluid, dt_wall)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt,
                      t=current_t, t_final=t_final,
                      #max_steps=niter, local_dt=local_dt,
                      #force_eval=force_eval,
                      state=stepper_state)

    current_cv, current_wv = stepper_state
    current_fluid_state = get_fluid_state(current_cv)
#XXX    current_wall_kappa, current_wall_temperature = \
#XXX        create_wall_derived_compiled(current_wv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(current_cv, gas_model)

    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

    my_write_viz(step=current_step, t=current_t, #dt=current_dt,
                 cv=current_state.cv, dv=current_state.dv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "burner_mix"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, restart_filename=restart_file)
