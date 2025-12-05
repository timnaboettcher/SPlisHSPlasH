#ifndef __Coupling_PIISPH_h__
#define __Coupling_PIISPH_h__

#include "SPlisHSPlasH/Common.h"
#include "TimeStepPIISPH.h"

namespace SPH
{
	/** \brief This class implements the coupling forces for porous flow as
	* proposed by Böttcher et al. [BWJB25].
	*
	* References:
	* - [BWJB25] Timna Böttcher, Lukas Westhofen, Stefan Rhys Jeske, and Jan Bender. 2025. Implicit Incompressible Porous Flow using SPH. ACM Trans. Graph. 44, 6, Article 267 (December 2025), 13 pages. https://doi.org/10.1145/3763325
	*/
	class Coupling_PIISPH
	{
	protected:
		TimeStepPIISPH* m_timeStep;

	public:
		Coupling_PIISPH(TimeStepPIISPH* timeStep);
		void computeForces(const unsigned int fluidModelIndex, const Real* vec);
		void computeRHS(VectorXr& b);
	};
}

#endif
