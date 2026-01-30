#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Range.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/BlockStructureDescriptor.h"
#include <complex>
using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;
complex<double> i(0, 1);


//Lattice size
const int SIZE_X = 100;
const int SIZE_Y = 10;

//Order parameter. 
Array<complex<double>> Delta({SIZE_X, SIZE_Y});

//Superconducting pair potential, convergence limit, max iterations, and initial guess
// and other Parameters.
const complex<double> mu = 0.0;
const complex<double> t = 1.0;
const double V_sc = 3.2;
const double J = 0.5;//1.00; // Zeeman field strength
const int MAX_ITERATIONS = 50;
const double CONVERGENCE_LIMIT = 0.000001;
const complex<double> DELTA_INITIAL_GUESS = 0.3 + 0.0*i;
const double DELTA_INITIAL_GUESS_RANDOM_WINDOW = 0.0;
const bool PERIODIC_BC_X = true;
const bool PERIODIC_BC_Y = true;
const bool SELF_CONSISTENCY = true;
const bool USE_GPU = false;
const bool USE_MULTI_GPU = false;

// double getParity(PropertyExtractor::Diagonalizer& pe, unsigned sizeX, unsigned sizeY){
// 	double n_up = 0;
// 	double n_down = 0;
// 	double h_up = 0;
// 	double h_down = 0;
// 	for(unsigned x = 0; x < sizeX; ++x){
// 		for(unsigned y = 0; y < sizeX; ++y){
			
// 			n_up += real(pe.calculateExpectationValue({0, 0, x, y}, {0, 0, x, y}));
// 			n_down += real(pe.calculateExpectationValue({1, 0, x, y}, {1, 0, x, y}));
// 			h_up += real(pe.calculateExpectationValue({0, 1, x, y}, {0, 1,x, y}));
// 			h_down += real(pe.calculateExpectationValue({1, 1, x, y}, {1, 1, x, y}));
// 		}
// 	}
// 	std::cout << "Nup: " << n_up << ", Ndown: " << n_down << std::endl;
// 	std::cout << "Hup: " << h_up << ", Hdown: " << h_down << std::endl;
// 	std::cout << "Total: " << h_up + h_down + n_up + n_down << std::endl;
// 	return n_up + n_down;
// }

bool selfConsistencyStep(Solver::Diagonalizer solver){
	PropertyExtractor::Diagonalizer property_extractor(solver);
	Array<complex<double>> delta_old = Delta;
	//Clear the order parameter of the next step
	for(unsigned int x = 0; x < SIZE_X; x++)
		for(unsigned int y = 0; y < SIZE_Y; y++)
			Delta[{x, y}] = 0.;


	//Calculate new order parameter from gap equation for each site
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned spin = 0; spin < 2; ++spin){
				// Gap equation
				Delta[{x, y}] += 0.5*V_sc *(-1.+2*spin) * property_extractor.calculateExpectationValue(
					{(spin+1)%2, 1, x,y},
					{spin, 0, x,y}
				);
			}
		}
	}
	//Calculate convergence parameter
	double maxError = 0.;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			double error = abs(Delta[{x,y}] - delta_old[{x,y}]);
			if(error > maxError)
				maxError = error;
		}
	}

	//Return true or false depending on whether the result has converged or not
	if(maxError < CONVERGENCE_LIMIT)
		return true;
	else
		return false;

}

//Callback function responsible for determining the value of the order
//parameter D_{to,from}c_{to}c_{from} where to and from are indices of the form
//(x, y, spin).
class DeltaCallback : public HoppingAmplitude::AmplitudeCallback{
	complex<double> getHoppingAmplitude(
		const Index &to,
		const Index &from
	) const{
		//Obtain indices
		unsigned int x = from[2];
		unsigned int y = from[3];
		unsigned int spin = from[0];
		unsigned int particleHole = from[1];

		if(spin == 0 && particleHole == 0)
			return conj(Delta[{x, y}]);
		else if(spin == 1 && particleHole == 0)
			return -conj(Delta[{x, y}]);
		else if(spin == 0 && particleHole == 1)
			return -Delta[{x, y}];
		else
			return Delta[{x, y}];
	}
} deltaCallback;

//Function responsible for initializing the order parameter
void initDelta(){
	const double rand_spread = DELTA_INITIAL_GUESS_RANDOM_WINDOW*abs(DELTA_INITIAL_GUESS);
	srand (static_cast <unsigned> (time(0)));
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			double a = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			double b = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			Delta[{x, y}] = DELTA_INITIAL_GUESS + rand_spread*(a+i*b);
		}
	}
}


//Callback that allows for the Zeeman term (J) to be updated after the Model
//has been set up.
class JCallback : public HoppingAmplitude::AmplitudeCallback{
public:
    //Function that returns the HoppingAmplitude value for the given
    //Indices. The to- and from-Indices are indentical in this example.
    complex<double> getHoppingAmplitude(
        const Index &to,
        const Index &from
    ) const{
        Subindex spin = from[0];
        Subindex particleHole = from[1];
        return J*(1. - 2*spin)*(1. - 2*particleHole);
    }
    //Set the value for J.
    void setJ(complex<double> J){
        this->J = J;
    }
private:
    complex<double> J;
};


// This function is supposed to divide the Hamiltonian into two blocks with the first block being particle spin up and hole spin down.
// The second block is the other combination 

int getBlockIndex(
    const Index &index) {
    Subindex spin = index[0];
    Subindex particleHole = index[1];

    if(spin == particleHole){
        return 0;
    }
    else{
        return 1;
    }
}


int main(){
    //Initialize TBTK.
    Initialize();
    //Parameters.
    const unsigned int SIZE_X = 11;
    const unsigned int SIZE_Y = 11;
    const double t = -1;
    const double mu = 0;
    const double Delta = 0.5;
    //Create a callback that returns the Zeeman term and that will be used
    //as input to the Model.
    JCallback jCallback;
    //Set up the Model.
    Model model;
    for(unsigned int x = 0; x < SIZE_X; x++){
        for(unsigned int y = 0; y < SIZE_Y; y++){
            for(unsigned int spin = 0; spin < 2; spin++){
                for(unsigned int ph = 0; ph < 2; ph++){
                    int block = getBlockIndex({spin, ph});
                    model << HoppingAmplitude(
                        -mu*(1. - 2*ph),
                        {block, spin, ph, x, y},
                        {block, spin, ph, x, y}
                    );
                    if(x+1 < SIZE_X){
                        model << HoppingAmplitude(
                            t*(1. - 2*ph),
                            {block, spin, ph, x+1, y},
                            {block, spin, ph, x, y}
                        ) + HC;
                    }
                    if(y+1 < SIZE_Y){
                        model << HoppingAmplitude(
                            t*(1. - 2*ph),
                            {block, spin, ph, x, y+1},
                            {block, spin, ph, x, y}
                        ) + HC;
                    }
                }
                int block_to = getBlockIndex({spin, 0});
                int block_from = getBlockIndex({(spin+1)%2, 1});
                model << HoppingAmplitude(
                    Delta*(1. - 2*spin),
                    {block_to, spin, 0, x, y},
                    {block_from, (spin+1)%2, 1, x, y}
                ) + HC;
            }
        }
    }
    for(unsigned int spin = 0; spin < 2; spin++){
        for(unsigned int ph = 0; ph < 2; ph++){
            int block = getBlockIndex({spin, ph});
            model << HoppingAmplitude(
                jCallback,
                {block, spin, ph, SIZE_X/2, SIZE_Y/2},
                {block, spin, ph, SIZE_X/2, SIZE_Y/2}
            );
        }
    }
    model.construct();
    //Number of iterations.
    const unsigned int NUM_ITERATIONS = 100;
    //Arrays where the results are stored after each iteration.
    Array<double> totalLdos({NUM_ITERATIONS, 500}, 0);
    Array<double> totalEigenValues({
        NUM_ITERATIONS,
        (unsigned int)model.getBasisSize()
    });
    //Iterate over 100 values for J.
    Range j(0, 5, NUM_ITERATIONS);


    // BlockStructureDescriptor blockStructureDescriptor = BlockStructureDescriptor(
	// 			model.getHoppingAmplitudeSet());
    // cout << blockStructureDescriptor.getNumBlocks() << endl;
    // exit(0);
    // Solver::BlockDiagonalizer blockSolver;
    // blockSolver.setModel(model);

    for(unsigned int n = 0; n < NUM_ITERATIONS; n++){
        //Update the callback with the current value of J.
        jCallback.setJ(j[n]);
        //Set up the Solver.
        Solver::Diagonalizer solver;
        solver.setModel(model);
        solver.run();
        //Set up the PropertyExtractor.
        PropertyExtractor::Diagonalizer propertyExtractor(solver);
        propertyExtractor.setEnergyWindow(
            -10,
            10,
            10000000
        );
		std::cout << "J: " << j[n] << std::endl;
		// getParity(propertyExtractor, SIZE_X, SIZE_Y) ;
        //Calculate the eigenvalues.
        Property::EigenValues eigenValues
            = propertyExtractor.getEigenValues();
        //Calculate the local density of states (LDOS).
        const double LOWER_BOUND = -5;
        const double UPPER_BOUND = 5;
        const unsigned int RESOLUTION = 500;
        propertyExtractor.setEnergyWindow(
            LOWER_BOUND,
            UPPER_BOUND,
            RESOLUTION
        );
        Property::LDOS ldos = propertyExtractor.calculateLDOS({
            {IDX_SUM_ALL, IDX_SUM_ALL, IDX_SUM_ALL, SIZE_X/2, SIZE_Y/2},
            {IDX_SUM_ALL, IDX_SUM_ALL, IDX_SUM_ALL, SIZE_X/4, SIZE_Y/4}
        });
        //Smooth the LDOS.
        const double SMOOTHING_SIGMA = 0.1;
        const unsigned int SMOOTHING_WINDOW = 51;
        ldos = Smooth::gaussian(
            ldos,
            SMOOTHING_SIGMA,
            SMOOTHING_WINDOW
        );
        //Store the LDOS in totalLdos.
        for(unsigned int e = 0; e < ldos.getResolution(); e++){
            totalLdos[{n, e}] = ldos(
                {
                    IDX_SUM_ALL,
                    IDX_SUM_ALL,
                    IDX_SUM_ALL,
                    SIZE_X/2,
                    SIZE_Y/2
                },
                e
            );
        }
        //Store the eigenvalues in totalEigenValues
        for(unsigned int e = 0; e < eigenValues.getSize(); e++)
            totalEigenValues[{n, e}] = eigenValues(e);
    }
    //Plot the LDOS.
    Plotter plotter;
    plotter.setNumContours(100);
    plotter.setAxes({
        {0, {0, 5}},
        {1, {-5, 5}},
    });
    plotter.setTitle("LDOS");
    plotter.setLabelX("J");
    plotter.setLabelY("Energy");
    plotter.setBoundsY(-5, 5);
    plotter.plot(totalLdos);
    plotter.save("LDOS.png");
    //Plot the eigenvalues.
    plotter.clear();
    plotter.setTitle("Eigenvalues");
    plotter.setLabelX("J");
    plotter.setLabelY("Energy");
    plotter.setAxes({
        {0, {0, 5}},
        {1, {-5, 5}}
    });
    plotter.setBoundsY(-5, 5);
    for(unsigned int e = 0; e < (unsigned int)model.getBasisSize(); e++){
        plotter.plot(
            totalEigenValues.getSlice({_a_, e}),
            {{"color", "black"}, {"linestyle", "-"}}
        );
    }
    plotter.save("EigenValues.png");
}