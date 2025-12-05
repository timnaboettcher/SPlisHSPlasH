#ifndef __SimulationDataPIISPH_h__
#define __SimulationDataPIISPH_h__

#include "SPlisHSPlasH/Common.h"
#include <vector>
#include "SPlisHSPlasH/FluidModel.h"

namespace SPH 
{	
	/** \brief Simulation data which is required by the porous flow method by Böttcher et al. [BWJB25].
	*
	* References:
	* - [BWJB25] Timna Böttcher, Lukas Westhofen, Stefan Rhys Jeske, and Jan Bender. 2025. Implicit Incompressible Porous Flow using SPH. ACM Trans. Graph. 44, 6, Article 267 (December 2025), 13 pages. https://doi.org/10.1145/3763325
	* - [BK15] Jan Bender and Dan Koschier. Divergence-free smoothed particle hydrodynamics. In ACM SIGGRAPH / Eurographics Symposium on Computer Animation, SCA '15, 147-155. New York, NY, USA, 2015. ACM. URL: http://doi.acm.org/10.1145/2786784.2786796
	* - [BK17] Jan Bender and Dan Koschier. Divergence-free SPH for incompressible and viscous fluids. IEEE Transactions on Visualization and Computer Graphics, 23(3):1193-1206, 2017. URL: http://dx.doi.org/10.1109/TVCG.2016.2578335
	*/
	class SimulationDataPIISPH
	{
		public:
			SimulationDataPIISPH();
			virtual ~SimulationDataPIISPH();

		protected:	
			/** \brief factor \f$\alpha_i\f$ */
			std::vector<std::vector<Real>> m_factor;
			/** \brief advected density */
			std::vector<std::vector<Real>> m_density_adv;

			/** \brief stores \f$\frac{p}{\rho^2}\f$ value of the constant density solver */
			std::vector<std::vector<Real>> m_pressure_rho2;
			std::vector<std::vector<Vector3r>> m_pressureAccel;
			std::vector<std::vector<Vector3r>> m_viscoForce;
			std::vector<std::vector<Vector3r>> m_elasticityForce;
			std::vector<std::vector<Vector3r>> m_couplingForce;

			std::vector<std::vector<Vector3r>> m_vDiff;

			// solids
			// initial particle indices, used to access their original positions
			std::vector<std::vector<Real>> m_saturation; // saturation (solid)
			std::vector<std::vector<Real>> m_porosity; // local porosity (fluid)
			std::vector<std::vector<unsigned int>> m_current_to_initial_index;
			std::vector<std::vector<unsigned int>> m_initial_to_current_index;
			// initial particle neighborhood
			std::vector < std::vector<std::vector<unsigned int>>> m_initialNeighbors;
			// volumes in rest configuration
			std::vector<std::vector<Real>> m_restVolumes;
			std::vector<std::vector<Matrix3r>> m_rotations;
			std::vector<std::vector<Matrix3r>> m_stress;
			std::vector<std::vector<Matrix3r>> m_L;
			std::vector<std::vector<Matrix3r>> m_RL;
			std::vector<std::vector<Matrix3r>> m_F;


		public:

			/** Initialize the arrays containing the particle data.
			*/
			virtual void init();

			/** Release the arrays containing the particle data.
			*/
			virtual void cleanup();

			/** Reset the particle data.
			*/
			virtual void reset();

			/** Important: First call m_model->performNeighborhoodSearchSort() 
			 * to call the z_sort of the neighborhood search.
			 */
			void performNeighborhoodSearchSort();
			void emittedParticles(FluidModel *model, const unsigned int startIndex);

			std::vector<std::vector<Real>>& getPressureRho2Data() { return m_pressure_rho2; }

			FORCE_INLINE const Real getFactor(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_factor[fluidIndex][i];
			}

			FORCE_INLINE Real& getFactor(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_factor[fluidIndex][i];
			}

			FORCE_INLINE void setFactor(const unsigned int fluidIndex, const unsigned int i, const Real p)
			{
				m_factor[fluidIndex][i] = p;
			}

			FORCE_INLINE const Real getDensityAdv(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_density_adv[fluidIndex][i];
			}

			FORCE_INLINE Real& getDensityAdv(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_density_adv[fluidIndex][i];
			}

			FORCE_INLINE void setDensityAdv(const unsigned int fluidIndex, const unsigned int i, const Real d)
			{
				m_density_adv[fluidIndex][i] = d;
			}

			FORCE_INLINE const Real getPressureRho2(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_pressure_rho2[fluidIndex][i];
			}

			FORCE_INLINE Real& getPressureRho2(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_pressure_rho2[fluidIndex][i];
			}

			FORCE_INLINE void setPressureRho2(const unsigned int fluidIndex, const unsigned int i, const Real p)
			{
				m_pressure_rho2[fluidIndex][i] = p;
			}

			FORCE_INLINE Vector3r& getPressureAccel(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_pressureAccel[fluidIndex][i];
			}

			FORCE_INLINE const Vector3r& getPressureAccel(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_pressureAccel[fluidIndex][i];
			}

			FORCE_INLINE void setPressureAccel(const unsigned int fluidIndex, const unsigned int i, const Vector3r& val)
			{
				m_pressureAccel[fluidIndex][i] = val;
			}
			FORCE_INLINE Vector3r& getViscoForce(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_viscoForce[fluidIndex][i];
			}

			FORCE_INLINE const Vector3r& getViscoForce(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_viscoForce[fluidIndex][i];
			}

			FORCE_INLINE void setViscoForce(const unsigned int fluidIndex, const unsigned int i, const Vector3r& val)
			{
				m_viscoForce[fluidIndex][i] = val;
			}

			FORCE_INLINE Vector3r& getElasticityForce(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_elasticityForce[fluidIndex][i];
			}

			FORCE_INLINE const Vector3r& getElasticityForce(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_elasticityForce[fluidIndex][i];
			}

			FORCE_INLINE void setElasticityForce(const unsigned int fluidIndex, const unsigned int i, const Vector3r& val)
			{
				m_elasticityForce[fluidIndex][i] = val;
			}

			FORCE_INLINE Vector3r& getCouplingForce(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_couplingForce[fluidIndex][i];
			}

			FORCE_INLINE const Vector3r& getCouplingForce(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_couplingForce[fluidIndex][i];
			}

			FORCE_INLINE void setCouplingForce(const unsigned int fluidIndex, const unsigned int i, const Vector3r& val)
			{
				m_couplingForce[fluidIndex][i] = val;
			}

			FORCE_INLINE const Real getSaturation(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_saturation[fluidIndex][i];
			}

			FORCE_INLINE Real& getSaturation(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_saturation[fluidIndex][i];
			}

			FORCE_INLINE void setSaturation(const unsigned int fluidIndex, const unsigned int i, const Real val)
			{
				m_saturation[fluidIndex][i] = val;
			}

			FORCE_INLINE const Real getPorosity(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_porosity[fluidIndex][i];
			}

			FORCE_INLINE Real& getPorosity(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_porosity[fluidIndex][i];
			}

			FORCE_INLINE void setPorosity(const unsigned int fluidIndex, const unsigned int i, const Real val)
			{
				m_porosity[fluidIndex][i] = val;
			}

			FORCE_INLINE Vector3r& getVDiff(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_vDiff[fluidIndex][i];
			}

			FORCE_INLINE const Vector3r& getVDiff(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_vDiff[fluidIndex][i];
			}

			FORCE_INLINE void setVDiff(const unsigned int fluidIndex, const unsigned int i, const Vector3r& val)
			{
				m_vDiff[fluidIndex][i] = val;
			}

			FORCE_INLINE const unsigned int getCurrentToInitialIndex(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_current_to_initial_index[fluidIndex][i];
			}

			FORCE_INLINE unsigned int& getCurrentToInitialIndex(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_current_to_initial_index[fluidIndex][i];
			}

			FORCE_INLINE void setCurrentToInitialIndex(const unsigned int fluidIndex, const unsigned int i, const unsigned int p)
			{
				m_current_to_initial_index[fluidIndex][i] = p;
			}

			FORCE_INLINE const unsigned int getInitialToCurrentIndex(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_initial_to_current_index[fluidIndex][i];
			}

			FORCE_INLINE unsigned int& getInitialToCurrentIndex(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_initial_to_current_index[fluidIndex][i];
			}

			FORCE_INLINE void setInitialToCurrentIndex(const unsigned int fluidIndex, const unsigned int i, const unsigned int p)
			{
				m_initial_to_current_index[fluidIndex][i] = p;
			}

			FORCE_INLINE std::vector<unsigned int>& getInitialToCurrentIndex(const unsigned int fluidIndex)
			{
				return m_initial_to_current_index[fluidIndex];
			}

			FORCE_INLINE const std::vector<unsigned int>& getInitialNeighbors(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_initialNeighbors[fluidIndex][i];
			}

			FORCE_INLINE std::vector<unsigned int>& getInitialNeighbors(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_initialNeighbors[fluidIndex][i];
			}

			FORCE_INLINE void setInitialNeighbors(const unsigned int fluidIndex, const unsigned int i, const std::vector<unsigned int>& p)
			{
				m_initialNeighbors[fluidIndex][i] = p;
			}

			FORCE_INLINE const Real getRestVolume(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_restVolumes[fluidIndex][i];
			}

			FORCE_INLINE Real& getRestVolume(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_restVolumes[fluidIndex][i];
			}

			FORCE_INLINE void setRestVolume(const unsigned int fluidIndex, const unsigned int i, const Real p)
			{
				m_restVolumes[fluidIndex][i] = p;
			}

			FORCE_INLINE const Matrix3r& getRotation(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_rotations[fluidIndex][i];
			}

			FORCE_INLINE Matrix3r& getRotation(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_rotations[fluidIndex][i];
			}

			FORCE_INLINE void setRotation(const unsigned int fluidIndex, const unsigned int i, const Matrix3r& p)
			{
				m_rotations[fluidIndex][i] = p;
			}

			FORCE_INLINE const Matrix3r& getStress(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_stress[fluidIndex][i];
			}

			FORCE_INLINE Matrix3r& getStress(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_stress[fluidIndex][i];
			}

			FORCE_INLINE void setStress(const unsigned int fluidIndex, const unsigned int i, const Matrix3r& p)
			{
				m_stress[fluidIndex][i] = p;
			}

			FORCE_INLINE const Matrix3r& getL(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_L[fluidIndex][i];
			}

			FORCE_INLINE Matrix3r& getL(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_L[fluidIndex][i];
			}

			FORCE_INLINE void setL(const unsigned int fluidIndex, const unsigned int i, const Matrix3r& p)
			{
				m_L[fluidIndex][i] = p;
			}

			FORCE_INLINE const Matrix3r& getRL(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_RL[fluidIndex][i];
			}

			FORCE_INLINE Matrix3r& getRL(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_RL[fluidIndex][i];
			}

			FORCE_INLINE void setRL(const unsigned int fluidIndex, const unsigned int i, const Matrix3r& p)
			{
				m_RL[fluidIndex][i] = p;
			}

			FORCE_INLINE const Matrix3r& getF(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_F[fluidIndex][i];
			}

			FORCE_INLINE Matrix3r& getF(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_F[fluidIndex][i];
			}

			FORCE_INLINE void setF(const unsigned int fluidIndex, const unsigned int i, const Matrix3r& p)
			{
				m_F[fluidIndex][i] = p;
			}

	};
}

#endif