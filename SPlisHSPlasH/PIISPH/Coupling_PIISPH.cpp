#include "Coupling_PIISPH.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataPIISPH.h"
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Simulation.h"
using namespace SPH;

Coupling_PIISPH::Coupling_PIISPH(TimeStepPIISPH* timeStep) : m_timeStep(timeStep)
{
}

#ifdef USE_AVX

void Coupling_PIISPH::computeForces(const unsigned int fluidModelIndex, const Real* vec)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);

	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();

	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	Real d = 10.0;
	if (sim->is2DSimulation())
		d = 8.0;
	const Real h = sim->getSupportRadius();
	const Real h2 = h * h;
	const Scalarf8 h2_001(0.01f * h2);

	const Real adhesion = m_timeStep->getAdhesion();
	const Real adhesion_mod = m_timeStep->getAdhesionFalloff();
	const Real drag = m_timeStep->getDrag();

	if (material->m_isFluid)
	{
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (model->getParticleState(i) != ParticleState::Active)
				{
					m_timeStep->getSimulationData().getCouplingForce(fluidModelIndex, i).setZero();
					continue;
				}

				const Vector3r& xi = model->getPosition(i);

				Vector3f8 force_avx;
				force_avx.setZero();

				const Real density_i = model->getDensity(i);
				const Vector3r& vi = Eigen::Map<const Vector3r>(&vec[3 * (i + m_timeStep->getBaseIndex(fluidModelIndex))]);

				const Vector3f8 xi_avx(xi);
				const Vector3f8 vi_avx(vi);
				const Scalarf8 density_i_avx(density_i);
				const Scalarf8 m_i_avx(model->getMass(i));
				const Scalarf8 phi_i_avx = Scalarf8(simulationData.getPorosity(fluidModelIndex,i));
				const Scalarf8 density_i_phi_avx = density_i_avx * phi_i_avx;

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors_avx_nox(
					compute_Vj(model);
					compute_Vj_gradW();
					if (!m_timeStep->getMaterialObject(pid)->m_isFluid)
					{
						const Vector3f8 xj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getPosition(0), count);
						const Scalarf8 m_j_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getMass(0), count);
						const Vector3f8 xixj = xi_avx - xj_avx;
						const Vector3f8 vj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &vec[3 * m_timeStep->getBaseIndex(pid)], count);
						const Scalarf8 density_j_avx(fm_neighbor->getDensity0());

						const Scalarf8 W_avx = CubicKernel_AVX::W(xixj);

						// drag
						if (drag > 0.0)
						{
							const Scalarf8 d_drag_rho0(d * drag * fm_neighbor->getDensity0()); // multiply with rho_0 to convert Vj_gradW into mj_gradV
							force_avx += V_gradW * m_i_avx / density_i_phi_avx * d_drag_rho0 / density_j_avx * (vi_avx - vj_avx).dot(xixj) / (xixj.squaredNorm() + h2_001);
						}
						// adhesion
						if (adhesion > 0.0)
						{
							const Scalarf8 saturation_j_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &simulationData.getSaturation(pid, 0), count);
							const Scalarf8 adh_dt_saturation_factor_avx = Scalarf8(adhesion * dt) * (Scalarf8(1.0) - saturation_j_avx * Scalarf8(adhesion_mod));
							const Scalarf8 rhoij = density_i_phi_avx + density_j_avx;
							const Scalarf8 m_ij = m_i_avx + m_j_avx;
							const Scalarf8 ones = ones_zero(count);
							force_avx -= (vi_avx - vj_avx) * ones * adh_dt_saturation_factor_avx * m_ij / rhoij * W_avx;
						}
					}
				);

				simulationData.getCouplingForce(fluidModelIndex, i) = force_avx.reduce();
			}
		}
	}
	else // solid
	{
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				Vector3r force;
				force.setZero();

				if (model->getParticleState(i) != ParticleState::Active)
				{
					m_timeStep->getSimulationData().getCouplingForce(fluidModelIndex, i).setZero();
					continue;
				}

				const Vector3r& xi = model->getPosition(i);

				Vector3f8 force_avx;
				force_avx.setZero();

				const Real density_i = model->getDensity(i);
				const Vector3r& vi = Eigen::Map<const Vector3r>(&vec[3 * (i + m_timeStep->getBaseIndex(fluidModelIndex))]);

				const Vector3f8 xi_avx(xi);
				const Vector3f8 vi_avx(vi);
				const Scalarf8 density_i_avx(density_i);
				const Scalarf8 m_i_avx(model->getMass(i));
				const Scalarf8 adh_dt_saturation_factor_avx(adhesion * dt * (static_cast<Real>(1.0) - simulationData.getSaturation(fluidModelIndex, i) * adhesion_mod));

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors_avx_nox(
					compute_Vj(model);
					compute_Vj_gradW();
					if (m_timeStep->getMaterialObject(pid)->m_isFluid)
					{
						const Vector3f8 xj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getPosition(0), count);
						const Scalarf8 m_j_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getMass(0), count);
						const Vector3f8 xixj = xi_avx - xj_avx;
						const Vector3f8 vj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &vec[3 * m_timeStep->getBaseIndex(pid)], count);
						const Scalarf8 density_j_avx = convert_one(&sim->getNeighborList(fluidModelIndex, fluidModelIndex, i)[j], &model->getDensity(0), count);
						const Scalarf8 phi_j_avx = convert_one(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &simulationData.getPorosity(pid,0), count);

						const Scalarf8 W_avx = CubicKernel_AVX::W(xixj);

						// drag
						if (drag > 0.0)
						{
							const Scalarf8 d_drag_rho0(d * drag * fm_neighbor->getDensity0()); // multiply with rho_0 to convert Vj_gradW into mj_gradV
							force_avx += V_gradW * m_i_avx / density_i_avx * d_drag_rho0 / (density_j_avx * phi_j_avx) * (vi_avx - vj_avx).dot(xixj) / (xixj.squaredNorm() + h2_001);
						}
						// adhesion
						if (adhesion > 0.0)
						{
							const Scalarf8 rhoij = density_i_avx + density_j_avx * phi_j_avx;
							const Scalarf8 m_ij = m_i_avx + m_j_avx;
							const Scalarf8 ones = ones_zero(count);
							force_avx -= (vi_avx - vj_avx) * ones * adh_dt_saturation_factor_avx * m_ij / rhoij * W_avx;
						}
					}
				);

				simulationData.getCouplingForce(fluidModelIndex, i) = force_avx.reduce();
			}
		}
	}
}

#else

void Coupling_PIISPH::computeForces(const unsigned int fluidModelIndex, const Real* vec)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);

	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();

	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
    Real d = 10.0;
    if (sim->is2DSimulation())
        d = 8.0;
    const Real h = sim->getSupportRadius();
    const Real h2 = h * h;

	const Real adhesion = m_timeStep->getAdhesion();
	const Real adhesion_mod = m_timeStep->getAdhesionFalloff();
	const Real drag = m_timeStep->getDrag();

	if (material->m_isFluid)
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				Vector3r force;
				force.setZero();

				const Vector3r& xi = model->getPosition(i);
				const Vector3r &vi = Eigen::Map<const Vector3r>(&vec[3 * (i + m_timeStep->getBaseIndex(fluidModelIndex))]);
				const Real& m_i = model->getMass(i);
				const Real& density_i = model->getDensity(i);
				const Real& phi_i = simulationData.getPorosity(fluidModelIndex,i);

				forall_fluid_neighbors(
					if (!m_timeStep->getMaterialObject(pid)->m_isFluid)
					{

						const Real density_j = fm_neighbor->getDensity0();
						const Real m_j = fm_neighbor->getMass(neighborIndex);
						const Vector3r xixj = xi - xj;

						const Vector3r &vj = Eigen::Map<const Vector3r>(&vec[3 * (neighborIndex + m_timeStep->getBaseIndex(pid))]);
						const Real W = sim->W(xixj);

						// adhesion
						if (adhesion > 0.0)
						{
							const Real rhoij = density_i * phi_i + density_j;
							const Real m_ij = m_i + m_j;
							const Real saturation_factor = 1.0 - simulationData.getSaturation(pid, neighborIndex) * adhesion_mod;
							force -= (vi - vj) * dt * saturation_factor * adhesion * m_ij / rhoij * W;
						}

						// drag
						if (drag > 0.0)
							force += d * drag * (m_j * m_i / (density_j * density_i * phi_i)) * (vi - vj).dot(xixj) / (xixj.squaredNorm() + 0.01*h2) * sim->gradW(xi - xj);
					}
				);

				simulationData.getCouplingForce(fluidModelIndex, i) = force;
			}
		}
	}
	else // solid
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				Vector3r force;
				force.setZero();

				const Vector3r& xi = model->getPosition(i);
				const Vector3r &vi = Eigen::Map<const Vector3r>(&vec[3 * (i + m_timeStep->getBaseIndex(fluidModelIndex))]);
				const Real& m_i = model->getMass(i);
				const Real& density_i = model->getDensity0();

				const Real saturation_factor = 1.0 - simulationData.getSaturation(fluidModelIndex,i) * adhesion_mod;

				forall_fluid_neighbors(
					if (m_timeStep->getMaterialObject(pid)->m_isFluid)
					{
						const Real density_j = fm_neighbor->getDensity(neighborIndex);
						const Real m_j = fm_neighbor->getMass(neighborIndex);
						const Vector3r xixj = xi - xj;

						const Vector3r &vj = Eigen::Map<const Vector3r>(&vec[3 * (neighborIndex + m_timeStep->getBaseIndex(pid))]);
						const Real Vj = m_j / density_j;

						const Real phi_j = simulationData.getPorosity(pid,neighborIndex);

						// adhesion
						if (adhesion > 0.0)
						{
							const Real rhoij = density_i + density_j * phi_j;
							const Real m_ij = m_i + m_j;
							force -= (vi - vj) * dt * saturation_factor * adhesion * m_ij / rhoij * sim->W(xixj);
						}
						// drag
						if (drag > 0.0)
							force += d * drag * (m_j * m_i / (density_j * phi_j * density_i)) * (vi - vj).dot(xixj) / (xixj.squaredNorm() + 0.01*h2) * sim->gradW(xi - xj);
						}
				);

				simulationData.getCouplingForce(fluidModelIndex, i) = force;
			}
		}
	}
}

#endif


#ifdef USE_AVX

void Coupling_PIISPH::computeRHS(VectorXr& b)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	const Real adhesion = m_timeStep->getAdhesion();
	const Real adhesion_mod = m_timeStep->getAdhesionFalloff();

#ifdef USE_WARMSTART
	// divide by 1 / dt^2 to remove warmstart correction
	const Real h2inv = static_cast<Real>(1.0) / (dt * dt);
#else
	const Real h2inv = static_cast<Real>(1.0);
#endif

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);

		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();

		if (material->m_isFluid && adhesion > 0.0)
		{
#pragma omp parallel default(shared)
			{
#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					Vector3r force;

					Vector3f8 force_avx;
					force_avx.setZero();

					const Vector3r& xi = model->getPosition(i);
					const Real density_i = model->getDensity(i);

					const Vector3f8 xi_avx(xi);
					const Scalarf8 density_i_avx(density_i);
					const Scalarf8 m_i_avx(model->getMass(i));
					const Scalarf8 phi_i_avx = Scalarf8(simulationData.getPorosity(fluidModelIndex, i));
					const Scalarf8 density_i_phi_avx = density_i_avx * phi_i_avx;

					{
						forall_fluid_neighbors_avx_nox(
							if (!m_timeStep->getMaterialObject(pid)->m_isFluid)
							{
								const Vector3f8 xj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getPosition(0), count);
								const Scalarf8 m_j_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getMass(0), count);
								const Vector3f8 xixj = xi_avx - xj_avx;
								const Scalarf8 density_j_avx(fm_neighbor->getDensity0());
								const Scalarf8 saturation_j_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &simulationData.getSaturation(pid, 0), count);

								// adhesion
								const Scalarf8 rhoij = density_i_phi_avx + density_j_avx;
								const Scalarf8 m_ij = m_i_avx + m_j_avx;
								const Scalarf8 W = m_ij * (CubicKernel_AVX::W(xixj) / rhoij);
								const Scalarf8 saturation_factor = static_cast<Real>(1.0) - saturation_j_avx * adhesion_mod;
								const Scalarf8 ones = ones_zero(count);
								force_avx -= xixj * ones * saturation_factor * Scalarf8(adhesion) * W;
							}
						)
					}

					force = force_avx.reduce();

					const unsigned int idx = 3 * (i + m_timeStep->getBaseIndex(fluidModelIndex));
					b[idx] += dt * force[0];
					b[idx + 1] += dt * force[1];
					b[idx + 2] += dt * force[2];
				}
			}
		}
		else if (!material->m_isFluid) // solid
		{
			const Real porosity0 = material->m_porosity;
			const Real density0 = model->getDensity0();

#pragma omp parallel default(shared)
			{
#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					Vector3r force;

					Vector3f8 force_avx;
					force_avx.setZero();

					const Scalarf8 saturation_factor(static_cast<Real>(1.0) - simulationData.getSaturation(fluidModelIndex, i) * adhesion_mod);
					const Vector3r& xi = model->getPosition(i);

					const Vector3f8 xi_avx(xi);
					const Scalarf8 density_i_avx(density0);
					const Scalarf8 m_i_avx(model->getMass(i));

					const Scalarf8 h2inv_por_Vi = (h2inv * (1 - porosity0) * model->getVolume(i));

					{
						forall_fluid_neighbors_avx_nox(
							if (m_timeStep->getMaterialObject(pid)->m_isFluid)
							{
								const Vector3f8 xj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getPosition(0), count);
								const Scalarf8 m_j_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &fm_neighbor->getMass(0), count);
								const Vector3f8 xixj = xi_avx - xj_avx;
								const Scalarf8 phi_j_avx = convert_one(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &simulationData.getPorosity(pid, 0), count);
								const Scalarf8 density_j_avx = convert_one(&sim->getNeighborList(fluidModelIndex, fluidModelIndex, i)[j], &model->getDensity(0), count);

								// adhesion
								if (adhesion > 0.0)
								{
									const Scalarf8 rhoij = density_i_avx + density_j_avx * phi_j_avx;
									const Scalarf8 m_ij = m_i_avx + m_j_avx;
									const Scalarf8 ones = ones_zero(count);
									const Scalarf8 W = m_ij * (CubicKernel_AVX::W(xixj) / rhoij);
									force_avx -= xixj * ones * saturation_factor * Scalarf8(adhesion) * W;
								}

								// buoyancy pressure force
								const Scalarf8 p_rho2_j = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &simulationData.getPressureRho2(pid, 0), count);
								force_avx -= CubicKernel_AVX::gradW(xixj) * m_j_avx * h2inv_por_Vi * p_rho2_j;
							}
						)
					}

					force = force_avx.reduce();

					const unsigned int idx = 3 * (i + m_timeStep->getBaseIndex(fluidModelIndex));
					b[idx] += dt * force[0];
					b[idx + 1] += dt * force[1];
					b[idx + 2] += dt * force[2];
				}
			}
		}
	}
}

#else

void Coupling_PIISPH::computeRHS(VectorXr& b)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	const Real adhesion = m_timeStep->getAdhesion();
	const Real adhesion_mod = m_timeStep->getAdhesionFalloff();

#ifdef USE_WARMSTART
	// divide by 1 / dt^2 to remove warmstart correction
	const Real h2inv = static_cast<Real>(1.0) / (dt * dt);
#else
	const Real h2inv = static_cast<Real>(1.0);
#endif

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
		
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();		
		
		if (material->m_isFluid)
		{
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					Vector3r force;
					force.setZero();

					const Vector3r& xi = model->getPosition(i);
					const Real& m_i = model->getMass(i);
					const Real& density_i = model->getDensity(i);
					const Real& phi_i = simulationData.getPorosity(fluidModelIndex,i);

					forall_fluid_neighbors(
						if (!m_timeStep->getMaterialObject(pid)->m_isFluid)
						{
							const Vector3r xixj = xi - xj;
							const Real m_j = fm_neighbor->getMass(neighborIndex);
							const Real density_j = fm_neighbor->getDensity0();
							const Real& saturation_j = simulationData.getSaturation(pid,neighborIndex);

							// adhesion
							const Real m_ij = m_i + m_j;
							const Real rhoij = density_i * phi_i + density_j;
							const Real W = m_ij * ( sim->W(xixj) / rhoij );
							const Real saturation_factor = static_cast<Real>(1.0) - saturation_j * adhesion_mod;
							force -= saturation_factor * adhesion * xixj * W;
						}
					)

					const unsigned int idx = 3 * (i + m_timeStep->getBaseIndex(fluidModelIndex));
					b[idx] += dt * force[0];
					b[idx + 1] += dt * force[1];
					b[idx + 2] += dt * force[2];
				}
			}
		}
		else // solid
		{
			const Real porosity0 = material->m_porosity;
			const Real density0 = model->getDensity0();

			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					Vector3r force;
					force.setZero();

					const Real saturation_factor = static_cast<Real>(1.0) - simulationData.getSaturation(fluidModelIndex,i) * adhesion_mod;
					const Vector3r& xi = model->getPosition(i);
					const Real& m_i = model->getMass(i);
					forall_fluid_neighbors(
						if (m_timeStep->getMaterialObject(pid)->m_isFluid)
						{
							const Vector3r xixj = xi - xj;
							const Real m_j = fm_neighbor->getMass(neighborIndex);
							const Real phi_j = simulationData.getPorosity(pid,neighborIndex);
							 
							const Real density_j = fm_neighbor->getDensity(neighborIndex);
							// adhesion
							if (adhesion > 0.0)
							{
								const Real m_ij = m_i + m_j;
								const Real rhoij = density0 + density_j * phi_j;
								const Real W = m_ij * ( sim->W(xixj) / rhoij );
								force -= saturation_factor * adhesion * xixj * W;
							}

							// buoyancy pressure force
							const Real p_rho2_j = simulationData.getPressureRho2(pid, neighborIndex)* h2inv;
							force -= m_j * (1 - porosity0) * p_rho2_j * sim->gradW(xixj) * model->getVolume(i);
						}
					)

					const unsigned int idx = 3 * (i + m_timeStep->getBaseIndex(fluidModelIndex));
					b[idx] += dt * force[0];
					b[idx + 1] += dt * force[1];
					b[idx + 2] += dt * force[2];
				}
			}
		}
	}
}

#endif