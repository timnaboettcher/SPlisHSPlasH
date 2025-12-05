#include "Viscosity_PIISPH.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataPIISPH.h"
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "SPlisHSPlasH/Utilities/MathFunctions.h"

using namespace SPH;

Viscosity_PIISPH::Viscosity_PIISPH(TimeStepPIISPH* timeStep) : m_timeStep(timeStep)
{
	m_tangentialDistanceFactor = static_cast<Real>(0.5);
}

#ifdef USE_AVX

void Viscosity_PIISPH::computeForces(const unsigned int fluidModelIndex, const Real* vec)
{
	PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
	const Real viscosity = material->m_viscosity;
	const Real viscosityBoundary = material->m_viscosityBoundary;
	if ((viscosity == 0.0) && (viscosityBoundary == 0.0))
		return;

	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

	const Real h = sim->getSupportRadius();
	const Real h2 = h * h;
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	const Real density0 = model->getDensity0();
	const Real mu = viscosity * density0;
	const Real mub = viscosityBoundary * density0;
	const Real sphereVolume = static_cast<Real>(4.0 / 3.0 * M_PI) * h2 * h;

	Real d = 10.0;
	if (sim->is2DSimulation())
		d = 8.0;

	const Scalarf8 d_mu_rho0(d * mu * density0);
	const Scalarf8 d_mub(d * mub);
	const Scalarf8 h2_001(0.01f * h2);
	const Scalarf8 density0_avx(density0);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static) 
		for (int i = 0; i < (int)numParticles; i++)
		{
			const Vector3r& xi = model->getPosition(i);
			Vector3r ai;
			ai.setZero();

			if (model->getParticleState(i) != ParticleState::Active)
			{
				m_timeStep->getSimulationData().getViscoForce(fluidModelIndex, i).setZero();
				continue;
			}

			const Real density_i = model->getDensity(i);
			const Vector3r& vi = Eigen::Map<const Vector3r>(&vec[3 * i]);

			const Vector3f8 xi_avx(xi);
			const Vector3f8 vi_avx(vi);
			const Scalarf8 density_i_avx(density_i);
			const Scalarf8 mi_avx(model->getMass(i));

			Vector3f8 delta_ai_avx;
			delta_ai_avx.setZero();

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			forall_fluid_neighbors_in_same_phase_avx(
				compute_Vj(model);
			compute_Vj_gradW_samephase();
			const Scalarf8 density_j_avx = convert_one(&sim->getNeighborList(fluidModelIndex, fluidModelIndex, i)[j], &model->getDensity(0), count);
			const Vector3f8 xixj = xi_avx - xj_avx;
			const Vector3f8 vj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, fluidModelIndex, i)[j], &vec[0], count);

			delta_ai_avx += V_gradW * ((d_mu_rho0 / density_j_avx) * (vi_avx - vj_avx).dot(xixj) / (xixj.squaredNorm() + h2_001));
			);

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			if (mub != 0.0)
			{
				if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
				{
					forall_boundary_neighbors_avx(
						const Vector3f8 vj_avx = convertVec_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &bm_neighbor->getVelocity(0), count);
					const Vector3f8 xixj = xi_avx - xj_avx;
					const Vector3f8 gradW = CubicKernel_AVX::gradW(xixj);
					const Scalarf8 Vj_avx = convert_zero(&sim->getNeighborList(fluidModelIndex, pid, i)[j], &bm_neighbor->getVolume(0), count);

					const Vector3f8 a = gradW * (d_mub * (density0_avx * Vj_avx / density_i_avx) * (vi_avx).dot(xixj) / (xixj.squaredNorm() + h2_001));
					delta_ai_avx += a;
					);
				}
				else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
				{
					forall_density_maps(
						const Vector3r xixj = xi - xj;
					Vector3r normal = -xixj;
					const Real nl = normal.norm();
					if (nl > static_cast<Real>(0.0001))
					{
						normal /= nl;

						Vector3r t1;
						Vector3r t2;
						MathFunctions::getOrthogonalVectors(normal, t1, t2);

						const Real dist = m_tangentialDistanceFactor * h;
						const Vector3r x1 = xj - t1 * dist;
						const Vector3r x2 = xj + t1 * dist;
						const Vector3r x3 = xj - t2 * dist;
						const Vector3r x4 = xj + t2 * dist;

						const Vector3r xix1 = xi - x1;
						const Vector3r xix2 = xi - x2;
						const Vector3r xix3 = xi - x3;
						const Vector3r xix4 = xi - x4;

						const Vector3r gradW1 = sim->gradW(xix1);
						const Vector3r gradW2 = sim->gradW(xix2);
						const Vector3r gradW3 = sim->gradW(xix3);
						const Vector3r gradW4 = sim->gradW(xix4);

						// each sample point represents the quarter of the volume inside of the boundary
						const Real vol = static_cast<Real>(0.25) * rho * sphereVolume;

						Vector3r v1;
						Vector3r v2;
						Vector3r v3;
						Vector3r v4;
						bm_neighbor->getPointVelocity(x1, v1);
						bm_neighbor->getPointVelocity(x2, v2);
						bm_neighbor->getPointVelocity(x3, v3);
						bm_neighbor->getPointVelocity(x4, v4);

						// compute forces for both sample point
						const Vector3r a1 = (d * mub * vol * (vi).dot(xix1) / (xix1.squaredNorm() + 0.01 * h2)) * gradW1;
						const Vector3r a2 = (d * mub * vol * (vi).dot(xix2) / (xix2.squaredNorm() + 0.01 * h2)) * gradW2;
						const Vector3r a3 = (d * mub * vol * (vi).dot(xix3) / (xix3.squaredNorm() + 0.01 * h2)) * gradW3;
						const Vector3r a4 = (d * mub * vol * (vi).dot(xix4) / (xix4.squaredNorm() + 0.01 * h2)) * gradW4;
						ai += a1 + a2 + a3 + a4;
					}
					);
				}
				else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
				{
					forall_volume_maps(
						const Vector3r xixj = xi - xj;
					Vector3r normal = -xixj;
					const Real nl = normal.norm();
					if (nl > static_cast<Real>(0.0001))
					{
						normal /= nl;

						Vector3r t1;
						Vector3r t2;
						MathFunctions::getOrthogonalVectors(normal, t1, t2);

						const Real dist = m_tangentialDistanceFactor * h;
						const Vector3r x1 = xj - t1 * dist;
						const Vector3r x2 = xj + t1 * dist;
						const Vector3r x3 = xj - t2 * dist;
						const Vector3r x4 = xj + t2 * dist;

						const Vector3r xix1 = xi - x1;
						const Vector3r xix2 = xi - x2;
						const Vector3r xix3 = xi - x3;
						const Vector3r xix4 = xi - x4;

						const Vector3r gradW1 = sim->gradW(xix1);
						const Vector3r gradW2 = sim->gradW(xix2);
						const Vector3r gradW3 = sim->gradW(xix3);
						const Vector3r gradW4 = sim->gradW(xix4);

						// each sample point represents the quarter of the volume inside of the boundary
						const Real vol = static_cast<Real>(0.25) * Vj;

						Vector3r v1;
						Vector3r v2;
						Vector3r v3;
						Vector3r v4;
						bm_neighbor->getPointVelocity(x1, v1);
						bm_neighbor->getPointVelocity(x2, v2);
						bm_neighbor->getPointVelocity(x3, v3);
						bm_neighbor->getPointVelocity(x4, v4);

						// compute forces for both sample point
						const Vector3r a1 = (d * mub * vol * (vi).dot(xix1) / (xix1.squaredNorm() + 0.01 * h2)) * gradW1;
						const Vector3r a2 = (d * mub * vol * (vi).dot(xix2) / (xix2.squaredNorm() + 0.01 * h2)) * gradW2;
						const Vector3r a3 = (d * mub * vol * (vi).dot(xix3) / (xix3.squaredNorm() + 0.01 * h2)) * gradW3;
						const Vector3r a4 = (d * mub * vol * (vi).dot(xix4) / (xix4.squaredNorm() + 0.01 * h2)) * gradW4;
						ai += a1 + a2 + a3 + a4;
					}
					);
				}
			}

			ai[0] += delta_ai_avx.x().reduce();
			ai[1] += delta_ai_avx.y().reduce();
			ai[2] += delta_ai_avx.z().reduce();

			m_timeStep->getSimulationData().getViscoForce(fluidModelIndex, i) = model->getMass(i) * ai / density_i;
		}
	}
}

#else

void Viscosity_PIISPH::computeForces(const unsigned int fluidModelIndex, const Real* vec)
{
	PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
	const Real viscosity = material->m_viscosity;
	const Real viscosityBoundary = material->m_viscosityBoundary;
	if ((viscosity == 0.0) && (viscosityBoundary == 0.0))
		return;

	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

	const Real h = sim->getSupportRadius();
	const Real h2 = h * h;
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	const Real density0 = model->getDensity0();
	const Real mu = viscosity * density0;
	const Real mub = viscosityBoundary * density0;
	const Real sphereVolume = static_cast<Real>(4.0 / 3.0 * M_PI) * h2 * h;

	Real d = 10.0;
	if (sim->is2DSimulation())
		d = 8.0;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static) 
		for (int i = 0; i < (int)numParticles; i++)
		{
			const Vector3r& xi = model->getPosition(i);
			Vector3r ai;
			ai.setZero();
			const Real density_i = model->getDensity(i);
			const Vector3r& vi = Eigen::Map<const Vector3r>(&vec[3 * i]);

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			forall_fluid_neighbors_in_same_phase(
				const Real density_j = model->getDensity(neighborIndex);
				const Vector3r gradW = sim->gradW(xi - xj);

				const Vector3r & vj = Eigen::Map<const Vector3r>(&vec[3 * neighborIndex]);
				const Vector3r xixj = xi - xj;

				ai += d * mu * (model->getMass(neighborIndex) / density_j) * (vi - vj).dot(xixj) / (xixj.squaredNorm() + 0.01 * h2) * gradW;
			);

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			if (mub != 0.0)
			{
				if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
				{
					forall_boundary_neighbors(
						const Vector3r & vj = bm_neighbor->getVelocity(neighborIndex);
					const Vector3r xixj = xi - xj;
					const Vector3r gradW = sim->gradW(xixj);
					const Vector3r a = d * mub * (density0 * bm_neighbor->getVolume(neighborIndex) / density_i) * (vi).dot(xixj) / (xixj.squaredNorm() + 0.01 * h2) * gradW;
					ai += a;
					);
				}
				else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
				{
					forall_density_maps(
						const Vector3r xixj = xi - xj;
					Vector3r normal = -xixj;
					const Real nl = normal.norm();
					if (nl > static_cast<Real>(0.0001))
					{
						normal /= nl;

						Vector3r t1;
						Vector3r t2;
						MathFunctions::getOrthogonalVectors(normal, t1, t2);

						const Real dist = m_tangentialDistanceFactor * h;
						const Vector3r x1 = xj - t1 * dist;
						const Vector3r x2 = xj + t1 * dist;
						const Vector3r x3 = xj - t2 * dist;
						const Vector3r x4 = xj + t2 * dist;

						const Vector3r xix1 = xi - x1;
						const Vector3r xix2 = xi - x2;
						const Vector3r xix3 = xi - x3;
						const Vector3r xix4 = xi - x4;

						const Vector3r gradW1 = sim->gradW(xix1);
						const Vector3r gradW2 = sim->gradW(xix2);
						const Vector3r gradW3 = sim->gradW(xix3);
						const Vector3r gradW4 = sim->gradW(xix4);

						// each sample point represents the quarter of the volume inside of the boundary
						const Real vol = static_cast<Real>(0.25) * rho * sphereVolume;

						Vector3r v1;
						Vector3r v2;
						Vector3r v3;
						Vector3r v4;
						bm_neighbor->getPointVelocity(x1, v1);
						bm_neighbor->getPointVelocity(x2, v2);
						bm_neighbor->getPointVelocity(x3, v3);
						bm_neighbor->getPointVelocity(x4, v4);

						// compute forces for both sample point
						const Vector3r a1 = d * mub * vol * (vi).dot(xix1) / (xix1.squaredNorm() + 0.01 * h2) * gradW1;
						const Vector3r a2 = d * mub * vol * (vi).dot(xix2) / (xix2.squaredNorm() + 0.01 * h2) * gradW2;
						const Vector3r a3 = d * mub * vol * (vi).dot(xix3) / (xix3.squaredNorm() + 0.01 * h2) * gradW3;
						const Vector3r a4 = d * mub * vol * (vi).dot(xix4) / (xix4.squaredNorm() + 0.01 * h2) * gradW4;
						ai += a1 + a2 + a3 + a4;
					}
					);
				}
				else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
				{
					forall_volume_maps(
						const Vector3r xixj = xi - xj;
					Vector3r normal = -xixj;
					const Real nl = normal.norm();
					if (nl > static_cast<Real>(0.0001))
					{
						normal /= nl;

						Vector3r t1;
						Vector3r t2;
						MathFunctions::getOrthogonalVectors(normal, t1, t2);

						const Real dist = m_tangentialDistanceFactor * h;
						const Vector3r x1 = xj - t1 * dist;
						const Vector3r x2 = xj + t1 * dist;
						const Vector3r x3 = xj - t2 * dist;
						const Vector3r x4 = xj + t2 * dist;

						const Vector3r xix1 = xi - x1;
						const Vector3r xix2 = xi - x2;
						const Vector3r xix3 = xi - x3;
						const Vector3r xix4 = xi - x4;

						const Vector3r gradW1 = sim->gradW(xix1);
						const Vector3r gradW2 = sim->gradW(xix2);
						const Vector3r gradW3 = sim->gradW(xix3);
						const Vector3r gradW4 = sim->gradW(xix4);

						// each sample point represents the quarter of the volume inside of the boundary
						const Real vol = static_cast<Real>(0.25) * Vj;

						Vector3r v1;
						Vector3r v2;
						Vector3r v3;
						Vector3r v4;
						bm_neighbor->getPointVelocity(x1, v1);
						bm_neighbor->getPointVelocity(x2, v2);
						bm_neighbor->getPointVelocity(x3, v3);
						bm_neighbor->getPointVelocity(x4, v4);

						// compute forces for both sample point
						const Vector3r a1 = d * mub * vol * (vi).dot(xix1) / (xix1.squaredNorm() + 0.01 * h2) * gradW1;
						const Vector3r a2 = d * mub * vol * (vi).dot(xix2) / (xix2.squaredNorm() + 0.01 * h2) * gradW2;
						const Vector3r a3 = d * mub * vol * (vi).dot(xix3) / (xix3.squaredNorm() + 0.01 * h2) * gradW3;
						const Vector3r a4 = d * mub * vol * (vi).dot(xix4) / (xix4.squaredNorm() + 0.01 * h2) * gradW4;
						ai += a1 + a2 + a3 + a4;
					}
					);
				}
			}

			m_timeStep->getSimulationData().getViscoForce(fluidModelIndex, i) = model->getMass(i) * ai / density_i;
		}
	}

}

#endif

void Viscosity_PIISPH::computeRHS(VectorXr& b)
{
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
		const Real viscosityBoundary = material->m_viscosityBoundary;
		if (viscosityBoundary == 0.0)
			continue;

		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();

		const Real h = sim->getSupportRadius();
		const Real h2 = h * h;
		const Real dt = TimeManager::getCurrent()->getTimeStepSize();
		const Real density0 = model->getDensity0();
		const Real mub = viscosityBoundary * density0;
		const Real sphereVolume = static_cast<Real>(4.0 / 3.0 * M_PI) * h2 * h;
		Real d = 10.0;
		if (sim->is2DSimulation())
			d = 8.0;

		//////////////////////////////////////////////////////////////////////////
		// Compute RHS
		//////////////////////////////////////////////////////////////////////////
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) nowait
			for (int i = 0; i < (int)numParticles; i++)
			{
				const Vector3r& vi = model->getVelocity(i);
				const Vector3r& xi = model->getPosition(i);
				const Real density_i = model->getDensity(i);
				const Real m_i = model->getMass(i);
				Vector3r bi = Vector3r::Zero();

				if (model->getParticleState(i) != ParticleState::Active)
					continue;

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				if (mub != 0.0)
				{
					if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
					{
						forall_boundary_neighbors(
							const Vector3r & vj = bm_neighbor->getVelocity(neighborIndex);
							const Vector3r xixj = xi - xj;
							const Vector3r gradW = sim->gradW(xixj);
							const Vector3r a = d * mub * (density0 * bm_neighbor->getVolume(neighborIndex) / density_i) * (vj).dot(xixj) / (xixj.squaredNorm() + 0.01 * h2) * gradW;
							bi += a;
						);
					}
					else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
					{
						forall_density_maps(
							const Vector3r xixj = xi - xj;
						Vector3r normal = -xixj;
						const Real nl = normal.norm();
						if (nl > static_cast<Real>(0.0001))
						{
							normal /= nl;

							Vector3r t1;
							Vector3r t2;
							MathFunctions::getOrthogonalVectors(normal, t1, t2);

							const Real dist = m_tangentialDistanceFactor * h;
							const Vector3r x1 = xj - t1 * dist;
							const Vector3r x2 = xj + t1 * dist;
							const Vector3r x3 = xj - t2 * dist;
							const Vector3r x4 = xj + t2 * dist;

							const Vector3r xix1 = xi - x1;
							const Vector3r xix2 = xi - x2;
							const Vector3r xix3 = xi - x3;
							const Vector3r xix4 = xi - x4;

							const Vector3r gradW1 = sim->gradW(xix1);
							const Vector3r gradW2 = sim->gradW(xix2);
							const Vector3r gradW3 = sim->gradW(xix3);
							const Vector3r gradW4 = sim->gradW(xix4);

							// each sample point represents the quarter of the volume inside of the boundary
							const Real vol = static_cast<Real>(0.25) * rho * sphereVolume;

							Vector3r v1;
							Vector3r v2;
							Vector3r v3;
							Vector3r v4;
							bm_neighbor->getPointVelocity(x1, v1);
							bm_neighbor->getPointVelocity(x2, v2);
							bm_neighbor->getPointVelocity(x3, v3);
							bm_neighbor->getPointVelocity(x4, v4);

							// compute forces for both sample point
							const Vector3r a1 = d * mub * vol * (v1).dot(xix1) / (xix1.squaredNorm() + 0.01 * h2) * gradW1;
							const Vector3r a2 = d * mub * vol * (v2).dot(xix2) / (xix2.squaredNorm() + 0.01 * h2) * gradW2;
							const Vector3r a3 = d * mub * vol * (v3).dot(xix3) / (xix3.squaredNorm() + 0.01 * h2) * gradW3;
							const Vector3r a4 = d * mub * vol * (v4).dot(xix4) / (xix4.squaredNorm() + 0.01 * h2) * gradW4;
							bi += a1 + a2 + a3 + a4;
						}
						);
					}
					else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
					{
						forall_volume_maps(
							const Vector3r xixj = xi - xj;
						Vector3r normal = -xixj;
						const Real nl = normal.norm();
						if (nl > static_cast<Real>(0.0001))
						{
							normal /= nl;

							Vector3r t1;
							Vector3r t2;
							MathFunctions::getOrthogonalVectors(normal, t1, t2);

							const Real dist = m_tangentialDistanceFactor * h;
							const Vector3r x1 = xj - t1 * dist;
							const Vector3r x2 = xj + t1 * dist;
							const Vector3r x3 = xj - t2 * dist;
							const Vector3r x4 = xj + t2 * dist;

							const Vector3r xix1 = xi - x1;
							const Vector3r xix2 = xi - x2;
							const Vector3r xix3 = xi - x3;
							const Vector3r xix4 = xi - x4;

							const Vector3r gradW1 = sim->gradW(xix1);
							const Vector3r gradW2 = sim->gradW(xix2);
							const Vector3r gradW3 = sim->gradW(xix3);
							const Vector3r gradW4 = sim->gradW(xix4);

							// each sample point represents the quarter of the volume inside of the boundary
							const Real vol = static_cast<Real>(0.25) * Vj;

							Vector3r v1;
							Vector3r v2;
							Vector3r v3;
							Vector3r v4;
							bm_neighbor->getPointVelocity(x1, v1);
							bm_neighbor->getPointVelocity(x2, v2);
							bm_neighbor->getPointVelocity(x3, v3);
							bm_neighbor->getPointVelocity(x4, v4);

							// compute forces for both sample point
							const Vector3r a1 = d * mub * vol * (v1).dot(xix1) / (xix1.squaredNorm() + 0.01 * h2) * gradW1;
							const Vector3r a2 = d * mub * vol * (v2).dot(xix2) / (xix2.squaredNorm() + 0.01 * h2) * gradW2;
							const Vector3r a3 = d * mub * vol * (v3).dot(xix3) / (xix3.squaredNorm() + 0.01 * h2) * gradW3;
							const Vector3r a4 = d * mub * vol * (v4).dot(xix4) / (xix4.squaredNorm() + 0.01 * h2) * gradW4;
							bi += a1 + a2 + a3 + a4;
						}
						);
					}
				}

				const unsigned int idx = 3 * (i + m_timeStep->getBaseIndex(fluidModelIndex));
				b[idx] += -model->getMass(i) * dt / density_i * bi[0];
				b[idx + 1] += -model->getMass(i) * dt / density_i * bi[1];
				b[idx + 2] += -model->getMass(i) * dt / density_i * bi[2];
			}
		}
	}

}