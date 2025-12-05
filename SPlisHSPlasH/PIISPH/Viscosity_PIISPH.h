#ifndef __Viscosity_PIISPH_h__
#define __Viscosity_PIISPH_h__

#include "SPlisHSPlasH/Common.h"
#include "TimeStepPIISPH.h"

namespace SPH
{
	/** \brief This class implements the implicit Laplace viscosity method introduced
	* by Weiler et al. 2018 [WKBB18], with modifications for porous flow as
	* proposed by Böttcher et al. [BWJB25].
	*
	* References:
	* - [BWJB25] Timna Böttcher, Lukas Westhofen, Stefan Rhys Jeske, and Jan Bender. 2025. Implicit Incompressible Porous Flow using SPH. ACM Trans. Graph. 44, 6, Article 267 (December 2025), 13 pages. https://doi.org/10.1145/3763325
	* - [WKBB18] Marcel Weiler, Dan Koschier, Magnus Brand, and Jan Bender. A physically consistent implicit viscosity solver for SPH fluids. Computer Graphics Forum (Eurographics), 2018. URL: https://doi.org/10.1111/cgf.13349
	*/
	class Viscosity_PIISPH
	{
	protected:
		TimeStepPIISPH* m_timeStep;
		Real m_tangentialDistanceFactor;

	public:
		Viscosity_PIISPH(TimeStepPIISPH* timeStep);
		void computeForces(const unsigned int fluidModelIndex, const Real* vec);
		void computeRHS(VectorXr& b);
	};
}

#endif
