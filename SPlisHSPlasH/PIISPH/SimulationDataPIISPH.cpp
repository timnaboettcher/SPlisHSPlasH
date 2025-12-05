#include "SimulationDataPIISPH.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SPlisHSPlasH/Simulation.h"

using namespace SPH;

SimulationDataPIISPH::SimulationDataPIISPH() :
	m_factor(),
	m_pressure_rho2(),
	m_pressureAccel(),
	m_density_adv(),
	m_viscoForce(),
	m_elasticityForce(),
	m_couplingForce(),
	m_saturation(),
	m_porosity()
{
}

SimulationDataPIISPH::~SimulationDataPIISPH(void)
{
	cleanup();
}

void SimulationDataPIISPH::init()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	m_factor.resize(nModels);
	m_density_adv.resize(nModels);
	m_pressure_rho2.resize(nModels);
	m_pressureAccel.resize(nModels);
	m_viscoForce.resize(nModels);
	m_elasticityForce.resize(nModels);
	m_couplingForce.resize(nModels);
	m_vDiff.resize(nModels);
	m_saturation.resize(nModels);
	m_porosity.resize(nModels);

	// solids
	m_current_to_initial_index.resize(nModels);
	m_initial_to_current_index.resize(nModels);
	m_initialNeighbors.resize(nModels);
	m_restVolumes.resize(nModels);
	m_rotations.resize(nModels);
	m_stress.resize(nModels);
	m_L.resize(nModels);
	m_RL.resize(nModels);
	m_F.resize(nModels);

	for (unsigned int i = 0; i < nModels; i++)
	{
		FluidModel *fm = sim->getFluidModel(i);
		m_factor[i].resize(fm->numParticles(), 0.0);
		m_density_adv[i].resize(fm->numParticles(), 0.0);
		m_pressure_rho2[i].resize(fm->numParticles(), 0.0);
		m_pressureAccel[i].resize(fm->numParticles(), Vector3r::Zero());
		m_viscoForce[i].resize(fm->numParticles(), Vector3r::Zero());
		m_elasticityForce[i].resize(fm->numParticles(), Vector3r::Zero());
		m_couplingForce[i].resize(fm->numParticles(), Vector3r::Zero());
		m_vDiff[i].resize(fm->numParticles(), Vector3r::Zero());
		m_saturation[i].resize(fm->numParticles(), 0.0);
		m_porosity[i].resize(fm->numParticles(), 1.0);

		// solids
		m_current_to_initial_index[i].resize(fm->numParticles(), 0);
		m_initial_to_current_index[i].resize(fm->numParticles(), 0);
		m_initialNeighbors[i].resize(fm->numParticles());
		m_restVolumes[i].resize(fm->numParticles(), 0.0);
		m_rotations[i].resize(fm->numParticles(), Matrix3r::Zero());
		m_stress[i].resize(fm->numParticles(), Matrix3r::Zero());
		m_L[i].resize(fm->numParticles(), Matrix3r::Zero());
		m_RL[i].resize(fm->numParticles(), Matrix3r::Zero());
		m_F[i].resize(fm->numParticles(), Matrix3r::Zero());
	}
}

void SimulationDataPIISPH::cleanup()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	for (unsigned int i = 0; i < nModels; i++)
	{
		m_factor[i].clear();
		m_density_adv[i].clear();
		m_pressure_rho2[i].clear();
		m_pressureAccel[i].clear();
		m_viscoForce[i].clear();
		m_elasticityForce[i].clear();
		m_couplingForce[i].clear();
		m_vDiff[i].clear();
		m_saturation[i].clear();
		m_porosity[i].clear();

		// solids
		m_current_to_initial_index[i].clear();
		m_initial_to_current_index[i].clear();
		m_initialNeighbors[i].clear();
		m_restVolumes[i].clear();
		m_rotations[i].clear();
		m_stress[i].clear();
		m_L[i].clear();
		m_RL[i].clear();
		m_F[i].clear();

	}
	m_factor.clear();
	m_density_adv.clear();
	m_pressure_rho2.clear();
	m_pressureAccel.clear();
	m_viscoForce.clear();
	m_elasticityForce.clear();
	m_couplingForce.clear();
	m_vDiff.clear();
	m_saturation.clear();
	m_porosity.clear();

	// solids
	m_current_to_initial_index.clear();
	m_initial_to_current_index.clear();
	m_initialNeighbors.clear();
	m_restVolumes.clear();
	m_rotations.clear();
	m_stress.clear();
	m_L.clear();
	m_RL.clear();
	m_F.clear();
}

void SimulationDataPIISPH::reset()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	for (unsigned int i = 0; i < nModels; i++)
	{
		FluidModel *fm = sim->getFluidModel(i);
		for (unsigned int j = 0; j < fm->numParticles(); j++)
		{
			m_density_adv[i][j] = 0.0;
			m_pressure_rho2[i][j] = 0.0;
			m_factor[i][j] = 0.0;
			m_pressureAccel[i][j].setZero();
			m_viscoForce[i][j].setZero();
			m_elasticityForce[i][j].setZero();
			m_couplingForce[i][j].setZero();
			m_saturation[i][j] = 0.0;
			m_porosity[i][j] = 1.0;
			m_vDiff[i][j].setZero();
			m_current_to_initial_index[i][j] = 0;
			m_initial_to_current_index[i][j] = 0;
			m_initialNeighbors[i][j].clear();
			m_restVolumes[i][j] = 0.0;
			m_rotations[i][j].setIdentity();
			m_stress[i][j].setZero();
			m_L[i][j].setZero();
			m_RL[i][j].setZero();
			m_F[i][j].setIdentity();
		}
	}
}

void SimulationDataPIISPH::performNeighborhoodSearchSort()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel* fm = sim->getFluidModel(fluidModelIndex);
		const unsigned int numPart = fm->numActiveParticles();
		if (numPart != 0)
		{
			auto const& d = sim->getNeighborhoodSearch()->point_set(fm->getPointSetIndex());
			d.sort_field(&m_pressure_rho2[fluidModelIndex][0]);
			d.sort_field(&m_vDiff[fluidModelIndex][0]);
			d.sort_field(&m_restVolumes[fluidModelIndex][0]);
			d.sort_field(&m_rotations[fluidModelIndex][0]);
			d.sort_field(&m_current_to_initial_index[fluidModelIndex][0]);
			d.sort_field(&m_L[fluidModelIndex][0]);

			for (unsigned int i = 0; i < numPart; i++)
				m_initial_to_current_index[fluidModelIndex][m_current_to_initial_index[fluidModelIndex][i]] = i;
		}
	}
}

void SimulationDataPIISPH::emittedParticles(FluidModel *model, const unsigned int startIndex)
{
	// initialize kappa values for new particles
	const unsigned int fluidModelIndex = model->getPointSetIndex();
	for (unsigned int j = startIndex; j < model->numActiveParticles(); j++)
	{
		m_pressure_rho2[fluidModelIndex][j] = 0.0;
		m_vDiff[fluidModelIndex][j].setZero();
	}
}
