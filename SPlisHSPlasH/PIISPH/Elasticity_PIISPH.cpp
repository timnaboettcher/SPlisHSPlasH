#include "Elasticity_PIISPH.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataPIISPH.h"
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Simulation.h"

using namespace SPH;

Elasticity_PIISPH::Elasticity_PIISPH(TimeStepPIISPH* timeStep) : m_timeStep(timeStep)
{
}

#ifdef USE_AVX

void Elasticity_PIISPH::computeForces(const unsigned int fluidModelIndex, const Real* vec)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
	const Real youngsModulus = material->m_youngsModulus;
	const Real poissonRatio = material->m_poissonRatio;

	if (youngsModulus == 0.0)
		return;

	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();

	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	Real mu = youngsModulus / (static_cast<Real>(2.0) * (static_cast<Real>(1.0) + poissonRatio));
	Real lambda = youngsModulus * poissonRatio / ((static_cast<Real>(1.0) + poissonRatio) * (static_cast<Real>(1.0) - static_cast<Real>(2.0) * poissonRatio));

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
			const Vector3r& pi = Eigen::Map<const Vector3r>(&vec[3 * i], 3);
			const Vector3r& xi0 = model->getPosition0(i0);

			std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
			const unsigned int numNeighbors = (unsigned int)initialNeighbors.size();

			//////////////////////////////////////////////////////////////////////////
			// compute corotated deformation gradient (Eq. 18)
			//////////////////////////////////////////////////////////////////////////
			Matrix3f8 nablaU_avx;
			nablaU_avx.setZero();
			const Vector3f8 pi_avx(pi);
			const Vector3f8 xi0_avx(xi0);
			const Matrix3f8 RLi(simulationData.getRL(fluidModelIndex, i));

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			for (unsigned int j = 0; j < numNeighbors; j += 8)
			{
				const unsigned int count = std::min(numNeighbors - j, 8u);
				const Vector3f8 xj0_avx = convertVec_zero(&initialNeighbors[j], &model->getPosition0(0), count);
				std::array<unsigned int, 8> indices;
				generateIndices(simulationData.getInitialToCurrentIndex(fluidModelIndex).data(), &initialNeighbors[j], indices, count);
				const Vector3f8 pj_avx = convertVec_zero(&indices[0], &vec[0], count);
				const Scalarf8 Vj0_avx = convert_zero(&indices[0], &simulationData.getRestVolume(fluidModelIndex, 0), count);

				const Vector3f8 pj_pi = pj_avx - pi_avx;
				const Vector3f8 xi_xj_0 = xi0_avx - xj0_avx;
				const Vector3f8 correctedRotatedKernel = RLi * CubicKernel_AVX::gradW(xi_xj_0);
				Matrix3f8 dyad;
				dyadicProduct(pj_pi, correctedRotatedKernel, dyad);
				nablaU_avx += dyad * Vj0_avx;
			}
			Matrix3r nablaU = nablaU_avx.reduce();
			nablaU *= dt;

			//////////////////////////////////////////////////////////////////////////
			// compute Cauchy strain: epsilon = 0.5 (nablaU + nablaU^T)
			//////////////////////////////////////////////////////////////////////////
			Vector6r strain;
			strain[0] = nablaU(0, 0);									// \epsilon_{00}
			strain[1] = nablaU(1, 1);									// \epsilon_{11}
			strain[2] = nablaU(2, 2);									// \epsilon_{22}
			strain[3] = static_cast<Real>(0.5) * (nablaU(0, 1) + nablaU(1, 0));			// \epsilon_{01}
			strain[4] = static_cast<Real>(0.5) * (nablaU(0, 2) + nablaU(2, 0));			// \epsilon_{02}
			strain[5] = static_cast<Real>(0.5) * (nablaU(1, 2) + nablaU(2, 1));			// \epsilon_{12}

			//////////////////////////////////////////////////////////////////////////
			// First Piola Kirchhoff stress = 2 mu epsilon + lambda trace(epsilon) I
			//////////////////////////////////////////////////////////////////////////

			Real sat = simulationData.getSaturation(fluidModelIndex, i);
			if (sat > 1.0) sat = 1.0;
			const Real trace = strain[0] + strain[1] + strain[2];

			const Real ltrace = fmax((static_cast<Real>(1.0) + sat * material->m_softeningVolume), static_cast<Real>(1e-3)) * trace;
			Real mu_i = fmax((static_cast<Real>(1.0) + sat * material->m_softeningShear), static_cast<Real>(1e-3)) * mu;

			Matrix3r& stress = simulationData.getStress(fluidModelIndex, i);

			stress(0, 0) = static_cast<Real>(2.0) * mu_i * strain[0] + ltrace;
			stress(1, 1) = static_cast<Real>(2.0) * mu_i * strain[1] + ltrace;
			stress(2, 2) = static_cast<Real>(2.0) * mu_i * strain[2] + ltrace;

			stress(0, 1) = static_cast<Real>(2.0) * mu_i * strain[3];
			stress(1, 0) = static_cast<Real>(2.0) * mu_i * strain[3];
			stress(0, 2) = static_cast<Real>(2.0) * mu_i * strain[4];
			stress(2, 0) = static_cast<Real>(2.0) * mu_i * strain[4];
			stress(1, 2) = static_cast<Real>(2.0) * mu_i * strain[5];
			stress(2, 1) = static_cast<Real>(2.0) * mu_i * strain[5];

		}
	}

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			if (model->getParticleState(i) == ParticleState::Active)
			{
				const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
				const Vector3r& xi0 = model->getPosition0(i0);

				std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
				const unsigned int numNeighbors = (unsigned int)initialNeighbors.size();

				//////////////////////////////////////////////////////////////////////////
				// Compute elastic force
				//////////////////////////////////////////////////////////////////////////
				Vector3f8 force_avx;
				force_avx.setZero();
				const Scalarf8 Vi0_avx(simulationData.getRestVolume(fluidModelIndex, i));
				const Vector3f8 xi0_avx(xi0);
				const Matrix3f8 RLi(simulationData.getRL(fluidModelIndex, i));
				const Matrix3f8 stress_i(simulationData.getStress(fluidModelIndex, i));
				for (unsigned int j = 0; j < numNeighbors; j += 8)
				{
					const unsigned int count = std::min(numNeighbors - j, 8u);

					std::array<unsigned int, 8> indices;
					generateIndices(simulationData.getInitialToCurrentIndex(fluidModelIndex).data(), &initialNeighbors[j], indices, count);

					const Matrix3f8& RLj = convertMat_zero(&indices[0], &simulationData.getRL(fluidModelIndex, 0), count);
					const Scalarf8 Vj0_avx = convert_zero(&indices[0], &simulationData.getRestVolume(fluidModelIndex, 0), count);
					const Vector3f8 xj0_avx = convertVec_zero(&initialNeighbors[j], &model->getPosition0(0), count);
					const Vector3f8 xi_xj_0 = xi0_avx - xj0_avx;
					const Vector3f8 gradW = CubicKernel_AVX::gradW(xi_xj_0);
					const Vector3f8 correctedRotatedKernel_i = RLi * gradW;
					const Vector3f8 correctedRotatedKernel_j = RLj * gradW;

					const Matrix3f8& stress_j = convertMat_zero(&indices[0], &simulationData.getStress(fluidModelIndex, 0), count);
					Vector3f8 PWi = stress_i * correctedRotatedKernel_i;
					Vector3f8 PWj = stress_j * correctedRotatedKernel_j;
					force_avx += (PWi + PWj) * Vi0_avx * Vj0_avx;
				}

				simulationData.getElasticityForce(fluidModelIndex, i) = force_avx.reduce();
			}
			else
			{
				simulationData.getElasticityForce(fluidModelIndex, i).setZero();
			}
		}
	}
}

#else

void Elasticity_PIISPH::computeForces(const unsigned int fluidModelIndex, const Real* vec)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
	const Real youngsModulus = material->m_youngsModulus;
	const Real poissonRatio = material->m_poissonRatio;

	if (youngsModulus == 0.0)
		return;

	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();

	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	Real mu = youngsModulus / (static_cast<Real>(2.0) * (static_cast<Real>(1.0) + poissonRatio));
	Real lambda = youngsModulus * poissonRatio / ((static_cast<Real>(1.0) + poissonRatio) * (static_cast<Real>(1.0) - static_cast<Real>(2.0) * poissonRatio));

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
			const Vector3r &pi = Eigen::Map<const Vector3r>(&vec[3 * i], 3);
			const Vector3r &xi0 = model->getPosition0(i0);
			const Matrix3r& RLi = simulationData.getRL(fluidModelIndex, i);
			std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
			const size_t numNeighbors = initialNeighbors.size();

 			//////////////////////////////////////////////////////////////////////////
 			// compute corotated deformation gradient (Eq. 18)
 			//////////////////////////////////////////////////////////////////////////
 			Matrix3r nablaU;
			nablaU.setZero();

  			//////////////////////////////////////////////////////////////////////////
 			// Fluid
 			//////////////////////////////////////////////////////////////////////////
 			for (unsigned int j = 0; j < numNeighbors; j++)
 			{

				const unsigned int neighborIndex = simulationData.getInitialToCurrentIndex(fluidModelIndex, initialNeighbors[j]);
 				// get initial neighbor index considering the current particle order 
 				const unsigned int neighborIndex0 = initialNeighbors[j];
 
 				const Vector3r &pj = Eigen::Map<const Vector3r>(&vec[3 * neighborIndex], 3);
 				const Vector3r &xj0 = model->getPosition0(neighborIndex0);
 				const Vector3r pj_pi = pj - pi;
 				const Vector3r xi_xj_0 = xi0 - xj0;
 				const Vector3r correctedRotatedKernel = RLi * sim->gradW(xi_xj_0);
				nablaU += simulationData.getRestVolume(fluidModelIndex, neighborIndex) * pj_pi * correctedRotatedKernel.transpose();
 			}
			nablaU *= dt;
 
 			//////////////////////////////////////////////////////////////////////////
 			// compute Cauchy strain: epsilon = 0.5 (nablaU + nablaU^T)
 			//////////////////////////////////////////////////////////////////////////

 			Vector6r strain;
			strain[0] = nablaU(0, 0);									// \epsilon_{00}
			strain[1] = nablaU(1, 1);									// \epsilon_{11}
			strain[2] = nablaU(2, 2);									// \epsilon_{22}
 			strain[3] = static_cast<Real>(0.5) * (nablaU(0, 1) + nablaU(1, 0));			// \epsilon_{01}
 			strain[4] = static_cast<Real>(0.5) * (nablaU(0, 2) + nablaU(2, 0));			// \epsilon_{02}
 			strain[5] = static_cast<Real>(0.5) * (nablaU(1, 2) + nablaU(2, 1));			// \epsilon_{12}

			//////////////////////////////////////////////////////////////////////////
			// First Piola Kirchhoff stress = 2 mu epsilon + lambda trace(epsilon) I
			//////////////////////////////////////////////////////////////////////////

			Real sat = simulationData.getSaturation(fluidModelIndex,i);
			if (sat > 1.0) sat = 1.0;
			const Real trace = strain[0] + strain[1] + strain[2];
			
			const Real ltrace = (static_cast<Real>(1.0) + sat * material->m_softeningVolume) * trace;
			const Real mu_i = (static_cast<Real>(1.0) + sat * material->m_softeningShear) * mu;

			Matrix3r& stress = simulationData.getStress(fluidModelIndex, i);
			stress(0, 0) = static_cast<Real>(2.0) * mu_i * strain[0] + ltrace;
			stress(1, 1) = static_cast<Real>(2.0) * mu_i * strain[1] + ltrace;
			stress(2, 2) = static_cast<Real>(2.0) * mu_i * strain[2] + ltrace;
			stress(0, 1) = static_cast<Real>(2.0) * mu_i * strain[3];
			stress(1, 0) = static_cast<Real>(2.0) * mu_i * strain[3];
			stress(0, 2) = static_cast<Real>(2.0) * mu_i * strain[4];
			stress(2, 0) = static_cast<Real>(2.0) * mu_i * strain[4];
			stress(1, 2) = static_cast<Real>(2.0) * mu_i * strain[5];
			stress(2, 1) = static_cast<Real>(2.0) * mu_i * strain[5];
		}
	}
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			if (model->getParticleState(i) == ParticleState::Active)
			{
				const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
				const Vector3r& xi0 = model->getPosition0(i0);
				
				std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
				const size_t numNeighbors = initialNeighbors.size();
				//////////////////////////////////////////////////////////////////////////
				// Compute elastic force
				//////////////////////////////////////////////////////////////////////////
				Vector3r force;
				force.setZero();
				const Matrix3r& RLi = simulationData.getRL(fluidModelIndex, i);
				const Matrix3r& stress_i = simulationData.getStress(fluidModelIndex, i);
				for (unsigned int j = 0; j < numNeighbors; j++)
				{
					const unsigned int neighborIndex = simulationData.getInitialToCurrentIndex(fluidModelIndex, initialNeighbors[j]);
					// get initial neighbor index considering the current particle order 
					const unsigned int neighborIndex0 = initialNeighbors[j];
					const Matrix3r& RLj = simulationData.getRL(fluidModelIndex, neighborIndex);

					const Vector3r& xj0 = model->getPosition0(neighborIndex0);
					const Vector3r xi_xj_0 = xi0 - xj0;
					const Vector3r gradW = sim->gradW(xi_xj_0);
					const Vector3r correctedRotatedKernel_i = RLi * gradW;
					const Vector3r correctedRotatedKernel_j = -RLj * gradW;
					const Matrix3r& stress_j = simulationData.getStress(fluidModelIndex, neighborIndex);
					const Vector3r PWi = stress_i * correctedRotatedKernel_i;
					const Vector3r PWj = stress_j * correctedRotatedKernel_j;
					force += simulationData.getRestVolume(fluidModelIndex, i) * simulationData.getRestVolume(fluidModelIndex, neighborIndex) * (PWi - PWj);
				}
				simulationData.getElasticityForce(fluidModelIndex, i) = force;
			}
			else
			{
				simulationData.getElasticityForce(fluidModelIndex, i).setZero();
			}
		}
	}
}

#endif

#ifdef USE_AVX

void Elasticity_PIISPH::computeRHS(VectorXr& b)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);

		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();

		const Real youngsModulus = material->m_youngsModulus;
		const Real poissonRatio = material->m_poissonRatio;

		if (youngsModulus == 0.0)
			continue;

		Real mu = youngsModulus / (static_cast<Real>(2.0) * (static_cast<Real>(1.0) + poissonRatio));
		Real lambda = youngsModulus * poissonRatio / ((static_cast<Real>(1.0) + poissonRatio) * (static_cast<Real>(1.0) - static_cast<Real>(2.0) * poissonRatio));

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
				const Vector3r& xi = model->getPosition(i);
				const Vector3r& xi0 = model->getPosition0(i0);
				std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
				const unsigned int numNeighbors = initialNeighbors.size();

				//////////////////////////////////////////////////////////////////////////
				// compute corotated deformation gradient (Eq. 18)
				//////////////////////////////////////////////////////////////////////////
				Matrix3f8 F_avx;
				F_avx.setZero();
				const Vector3f8 xi_avx(xi);
				const Vector3f8 xi0_avx(xi0);
				const Matrix3f8 Ri = simulationData.getRotation(fluidModelIndex, i);
				const Matrix3f8 RLi(simulationData.getRL(fluidModelIndex, i));

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < numNeighbors; j += 8)
				{
					const unsigned int count = std::min(numNeighbors - j, 8u);
					std::array<unsigned int, 8> indices;
					generateIndices(simulationData.getInitialToCurrentIndex(fluidModelIndex).data(), &initialNeighbors[j], indices, count);

					const Vector3f8 xj_avx = convertVec_zero(&indices[0], &model->getPosition(0), count);
					const Scalarf8 Vj0_avx = convert_zero(&indices[0], &simulationData.getRestVolume(fluidModelIndex, 0), count);
					const Vector3f8 xj0_avx = convertVec_zero(&initialNeighbors[j], &model->getPosition0(0), count);

					const Vector3f8 xj_xi = xj_avx - xi_avx;
					const Vector3f8 xi_xj_0 = xi0_avx - xj0_avx;
					const Vector3f8 correctedRotatedKernel = RLi * CubicKernel_AVX::gradW(xi_xj_0);

					Matrix3f8 dyad;
					dyadicProduct((xj_xi - Ri * (xj0_avx - xi0_avx)), correctedRotatedKernel, dyad);
					F_avx += dyad * Vj0_avx;
				}

				Matrix3r& F = simulationData.getF(fluidModelIndex, i);
				F = F_avx.reduce();

				F += Matrix3r::Identity();

				if (sim->is2DSimulation())
					F(2, 2) = 1.0;

				//////////////////////////////////////////////////////////////////////////
				// compute Cauchy strain: epsilon = 0.5 (F + F^T) - I
				Vector6r strain;
				strain[0] = F(0, 0) - static_cast<Real>(1.0);						// \epsilon_{00}
				strain[1] = F(1, 1) - static_cast<Real>(1.0);						// \epsilon_{11}
				strain[2] = F(2, 2) - static_cast<Real>(1.0);						// \epsilon_{22}
				strain[3] = static_cast<Real>(0.5) * (F(0, 1) + F(1, 0));			// \epsilon_{01}
				strain[4] = static_cast<Real>(0.5) * (F(0, 2) + F(2, 0));			// \epsilon_{02}
				strain[5] = static_cast<Real>(0.5) * (F(1, 2) + F(2, 1));			// \epsilon_{12}

				//////////////////////////////////////////////////////////////////////////
				// First Piola Kirchhoff stress = 2 mu epsilon + lambda trace(epsilon) I
				//////////////////////////////////////////////////////////////////////////

				Real saturation = simulationData.getSaturation(fluidModelIndex, i);
				if (saturation > 1.0) saturation = 1.0;
				const Real trace = strain[0] + strain[1] + strain[2];
				const Real ltrace = fmax(static_cast<Real>(1.0) + saturation * material->m_softeningVolume, static_cast<Real>(1e-3)) * trace - lambda * material->m_bloating * saturation;
				const Real mu_i = fmax(static_cast<Real>(1.0) + saturation * material->m_softeningShear, static_cast<Real>(1e-3)) * mu;

				Matrix3r& stress = simulationData.getStress(fluidModelIndex, i);
				stress(0, 0) = static_cast<Real>(2.0) * mu_i * strain[0] + ltrace;
				stress(1, 1) = static_cast<Real>(2.0) * mu_i * strain[1] + ltrace;
				stress(2, 2) = static_cast<Real>(2.0) * mu_i * strain[2] + ltrace;

				stress(0, 1) = static_cast<Real>(2.0) * mu_i * strain[3];
				stress(1, 0) = static_cast<Real>(2.0) * mu_i * strain[3];
				stress(0, 2) = static_cast<Real>(2.0) * mu_i * strain[4];
				stress(2, 0) = static_cast<Real>(2.0) * mu_i * strain[4];
				stress(1, 2) = static_cast<Real>(2.0) * mu_i * strain[5];
				stress(2, 1) = static_cast<Real>(2.0) * mu_i * strain[5];

			}
		}

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (model->getParticleState(i) == ParticleState::Active)
				{
					const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
					const Vector3r& xi0 = model->getPosition0(i0);

					std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
					const unsigned int numNeighbors = (unsigned int)initialNeighbors.size();

					//////////////////////////////////////////////////////////////////////////
					// Compute elastic force
					//////////////////////////////////////////////////////////////////////////
					Vector3f8 force_avx;
					force_avx.setZero();
					const Scalarf8 Vi0_avx(simulationData.getRestVolume(fluidModelIndex, i));
					const Vector3f8 xi0_avx(xi0);
					const Matrix3f8 RLi(simulationData.getRL(fluidModelIndex, i));
					const Matrix3f8 stress_i(simulationData.getStress(fluidModelIndex, i));
					for (unsigned int j = 0; j < numNeighbors; j += 8)
					{
						const unsigned int count = std::min(numNeighbors - j, 8u);

						std::array<unsigned int, 8> indices;
						generateIndices(simulationData.getInitialToCurrentIndex(fluidModelIndex).data(), &initialNeighbors[j], indices, count);

						const Matrix3f8& RLj = convertMat_zero(&indices[0], &simulationData.getRL(fluidModelIndex, 0), count);
						const Scalarf8 Vj0_avx = convert_zero(&indices[0], &simulationData.getRestVolume(fluidModelIndex, 0), count);
						const Vector3f8 xj0_avx = convertVec_zero(&initialNeighbors[j], &model->getPosition0(0), count);
						const Vector3f8 xi_xj_0 = xi0_avx - xj0_avx;
						const Vector3f8 gradW = CubicKernel_AVX::gradW(xi_xj_0);
						const Vector3f8 correctedRotatedKernel_i = RLi * gradW;
						const Vector3f8 correctedRotatedKernel_j = RLj * gradW;

						const Matrix3f8& stress_j = convertMat_zero(&indices[0], &simulationData.getStress(fluidModelIndex, 0), count);
						Vector3f8 PWi = stress_i * correctedRotatedKernel_i;
						Vector3f8 PWj = stress_j * correctedRotatedKernel_j;
						force_avx += (PWi + PWj) * Vi0_avx * Vj0_avx;
					}

					const Vector3r force = force_avx.reduce();
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

void Elasticity_PIISPH::computeRHS(VectorXr& b)
{
	SimulationDataPIISPH& simulationData = m_timeStep->getSimulationData();
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		PIISPHMaterialParameterObject* material = m_timeStep->getMaterialObject(fluidModelIndex);
		
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();		
		
		const Real youngsModulus = material->m_youngsModulus;
		const Real poissonRatio = material->m_poissonRatio;

		if (youngsModulus == 0.0)
			continue;

		Real mu = youngsModulus / (static_cast<Real>(2.0) * (static_cast<Real>(1.0) + poissonRatio));
		Real lambda = youngsModulus * poissonRatio / ((static_cast<Real>(1.0) + poissonRatio) * (static_cast<Real>(1.0) - static_cast<Real>(2.0) * poissonRatio));

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (model->getParticleState(i) == ParticleState::Active)
				{
					const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
					const Vector3r& xi = model->getPosition(i);
					const Vector3r& xi0 = model->getPosition0(i0);
					std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
					const size_t numNeighbors = initialNeighbors.size();

					//////////////////////////////////////////////////////////////////////////
					// compute corotated deformation gradient (Eq. 18)
					//////////////////////////////////////////////////////////////////////////
					Matrix3r& F = simulationData.getF(fluidModelIndex, i);
					F.setZero();
					const Matrix3r& Ri = simulationData.getRotation(fluidModelIndex, i);
					const Matrix3r& RLi = simulationData.getRL(fluidModelIndex, i);

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < numNeighbors; j++)
					{
						const unsigned int neighborIndex = simulationData.getInitialToCurrentIndex(fluidModelIndex, initialNeighbors[j]);
						// get initial neighbor index considering the current particle order 
						const unsigned int neighborIndex0 = initialNeighbors[j];

						const Vector3r& xj = model->getPosition(neighborIndex);
						const Vector3r& xj0 = model->getPosition0(neighborIndex0);
						const Vector3r xj_xi = xj - xi;
						const Vector3r xi_xj_0 = xi0 - xj0;
						const Vector3r correctedRotatedKernel = RLi * sim->gradW(xi_xj_0);
						F += simulationData.getRestVolume(fluidModelIndex, neighborIndex) * (xj_xi - Ri * (xj0 - xi0)) * correctedRotatedKernel.transpose();
					}

					F += Matrix3r::Identity();

					if (sim->is2DSimulation())
						F(2, 2) = 1.0;

					//////////////////////////////////////////////////////////////////////////
					// compute Cauchy strain: epsilon = 0.5 (F + F^T) - I
					Vector6r strain;
					strain[0] = F(0, 0) - static_cast<Real>(1.0);						// \epsilon_{00}
					strain[1] = F(1, 1) - static_cast<Real>(1.0);						// \epsilon_{11}
					strain[2] = F(2, 2) - static_cast<Real>(1.0);						// \epsilon_{22}
					strain[3] = static_cast<Real>(0.5) * (F(0, 1) + F(1, 0));			// \epsilon_{01}
					strain[4] = static_cast<Real>(0.5) * (F(0, 2) + F(2, 0));			// \epsilon_{02}
					strain[5] = static_cast<Real>(0.5) * (F(1, 2) + F(2, 1));			// \epsilon_{12}

					//////////////////////////////////////////////////////////////////////////
					// First Piola Kirchhoff stress = 2 mu epsilon + lambda trace(epsilon) I
					//////////////////////////////////////////////////////////////////////////
					Real saturation = simulationData.getSaturation(fluidModelIndex, i);
					if (saturation > 1.0) saturation = 1.0;
					const Real trace = strain[0] + strain[1] + strain[2];
					const Real ltrace = fmax(static_cast<Real>(1.0) + saturation * material->m_softeningVolume, static_cast<Real>(1e-3)) * trace - lambda * material->m_bloating * saturation;
					const Real mu_i = fmax(static_cast<Real>(1.0) + saturation * material->m_softeningShear, static_cast<Real>(1e-3)) * mu;

					Matrix3r& stress = simulationData.getStress(fluidModelIndex, i);
					stress(0, 0) = static_cast<Real>(2.0) * mu_i * strain[0] + ltrace;
					stress(1, 1) = static_cast<Real>(2.0) * mu_i * strain[1] + ltrace;
					stress(2, 2) = static_cast<Real>(2.0) * mu_i * strain[2] + ltrace;

					stress(0, 1) = static_cast<Real>(2.0) * mu_i * strain[3];
					stress(1, 0) = static_cast<Real>(2.0) * mu_i * strain[3];
					stress(0, 2) = static_cast<Real>(2.0) * mu_i * strain[4];
					stress(2, 0) = static_cast<Real>(2.0) * mu_i * strain[4];
					stress(1, 2) = static_cast<Real>(2.0) * mu_i * strain[5];
					stress(2, 1) = static_cast<Real>(2.0) * mu_i * strain[5];
				}
			}
		}

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (model->getParticleState(i) == ParticleState::Active)
				{
					const unsigned int i0 = simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
					const Vector3r& xi0 = model->getPosition0(i0);

					std::vector<unsigned int>& initialNeighbors = simulationData.getInitialNeighbors(fluidModelIndex, i0);
					const size_t numNeighbors = initialNeighbors.size();

					//////////////////////////////////////////////////////////////////////////
					// Compute elastic force
					//////////////////////////////////////////////////////////////////////////
					Vector3r force;
					force.setZero();
					const Matrix3r& RLi = simulationData.getRL(fluidModelIndex, i);
					const Matrix3r& stress_i = simulationData.getStress(fluidModelIndex, i);

					for (unsigned int j = 0; j < numNeighbors; j++)
					{
						const unsigned int neighborIndex = simulationData.getInitialToCurrentIndex(fluidModelIndex, initialNeighbors[j]);
						// get initial neighbor index considering the current particle order 
						const unsigned int neighborIndex0 = initialNeighbors[j];

						const Matrix3r& RLj = simulationData.getRL(fluidModelIndex, neighborIndex);

						const Vector3r& xj0 = model->getPosition0(neighborIndex0);
						const Vector3r xi_xj_0 = xi0 - xj0;
						const Vector3r correctedRotatedKernel_i = RLi * sim->gradW(xi_xj_0);
						const Vector3r correctedRotatedKernel_j = -RLj * sim->gradW(xi_xj_0);
						Vector3r PWi, PWj;
						const Matrix3r& stress_j = simulationData.getStress(fluidModelIndex, neighborIndex);
						PWi = stress_i * correctedRotatedKernel_i;
						PWj = stress_j * correctedRotatedKernel_j;
						force += simulationData.getRestVolume(fluidModelIndex, i) * simulationData.getRestVolume(fluidModelIndex, neighborIndex) * (PWi - PWj);

					}

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