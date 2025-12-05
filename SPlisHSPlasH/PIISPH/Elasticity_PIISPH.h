#ifndef __Elasticity_PIISPH_h__
#define __Elasticity_PIISPH_h__

#include "SPlisHSPlasH/Common.h"
#include "TimeStepPIISPH.h"

namespace SPH
{
	/** \brief This class implements the implicit SPH formulation for 
	* incompressible linearly elastic solids introduced
	* by Peer et al. [PGBT18], with modifications for porous flow as
	* proposed by Böttcher et al. [BWJB25].
	*
	* References:
	* - [BWJB25] Timna Böttcher, Lukas Westhofen, Stefan Rhys Jeske, and Jan Bender. 2025. Implicit Incompressible Porous Flow using SPH. ACM Trans. Graph. 44, 6, Article 267 (December 2025), 13 pages. https://doi.org/10.1145/3763325
	* - [PGBT18] Andreas Peer, Christoph Gissler, Stefan Band, and Matthias Teschner. An implicit SPH formulation for incompressible linearly elastic solids. Computer Graphics Forum, 2018. URL: http://dx.doi.org/10.1111/cgf.13317
	*/
	class Elasticity_PIISPH
	{
	protected:
		TimeStepPIISPH* m_timeStep;

		FORCE_INLINE void generateIndices(const unsigned int* map, const unsigned int* idx, std::array<unsigned int, 8>& indices, const unsigned char count = 8u)
		{
			switch (count)
			{
			case 1u:
				indices = { map[idx[0]], 0, 0, 0, 0, 0, 0, 0 }; break;
			case 2u:
				indices = { map[idx[0]], map[idx[1]], 0, 0, 0, 0, 0, 0 }; break;
			case 3u:
				indices = { map[idx[0]], map[idx[1]], map[idx[2]], 0, 0, 0, 0, 0 }; break;
			case 4u:
				indices = { map[idx[0]], map[idx[1]], map[idx[2]], map[idx[3]], 0, 0, 0, 0 }; break;
			case 5u:
				indices = { map[idx[0]], map[idx[1]], map[idx[2]], map[idx[3]], map[idx[4]], 0, 0, 0 }; break;
			case 6u:
				indices = { map[idx[0]], map[idx[1]], map[idx[2]], map[idx[3]], map[idx[4]], map[idx[5]], 0, 0 }; break;
			case 7u:
				indices = { map[idx[0]], map[idx[1]], map[idx[2]], map[idx[3]], map[idx[4]], map[idx[5]], map[idx[6]], 0 }; break;
			case 8u:
				indices = { map[idx[0]], map[idx[1]], map[idx[2]], map[idx[3]], map[idx[4]], map[idx[5]], map[idx[6]], map[idx[7]] }; break;
			}
		}

	public:
		Elasticity_PIISPH(TimeStepPIISPH* timeStep);
		void computeForces(const unsigned int fluidModelIndex, const Real* vec);
		void computeRHS(VectorXr& b);
	};
}

#endif
