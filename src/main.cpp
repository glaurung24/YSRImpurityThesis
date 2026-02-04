#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/Range.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Solver/Solver.h"
#include "TBTK/BlockStructureDescriptor.h"
#include "TBTK/Timer.h"
#include "TBTK/Index.h"
#include <complex>
#include "TBTK/Exporter.h"
using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;
complex<double> i(0, 1);


//Lattice size
const unsigned int SIZE = 31;
const unsigned int SIZE_X = SIZE;
const unsigned int SIZE_Y = SIZE;
const unsigned int SIZE_Z = SIZE;

//Order parameter. 
Array<complex<double>> Delta({SIZE_X, SIZE_Y});

//Superconducting pair potential, convergence limit, max iterations, and initial guess
// and other Parameters.
const complex<double> mu = 0.0;
const complex<double> t = -1.0;
const double V_sc = 3.2;
const double J = 1.0;//1.00; // Zeeman field strength
const double U = 0.0; //Local non magnetic scattering
const int MAX_ITERATIONS = 50;
const double CONVERGENCE_LIMIT = 0.000001;
const double DELTA = 0.3; // Non self consistent delta
const complex<double> DELTA_INITIAL_GUESS = 0.3 + 0.0*i;
const double DELTA_INITIAL_GUESS_RANDOM_WINDOW = 0.0;
const bool PERIODIC_BC_X = true;
const bool PERIODIC_BC_Y = true;
const bool SELF_CONSISTENCY = false;
const bool USE_GPU = true;
const bool USE_MULTI_GPU = true;
const unsigned int dimensions = 2;

const int useOnlyOneBlock = 0; // Set to 1 for first block, 2 for second block, 0 to use both


// Calculation details

const string DATA_DIR = "../../../../data/";
const string FIG_DIR = "../../../../figures/";

// This function is supposed to divide the Hamiltonian into two blocks with the first block being particle spin up and hole spin down.
// The second block is the other combination 

int getBlockIndex(
    const Index &index) {
    Subindex spin = index[0];
    Subindex particleHole = index[1];

    if(spin == particleHole){
        if(useOnlyOneBlock == 2) return -1; //Don't use this block
        return 0;
    }
    else{
        if(useOnlyOneBlock == 1) return -1; //Don't use this block
        return 1;
    }
}


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
                int block_to = getBlockIndex({spin, 0});
                int block_from = getBlockIndex({(spin+1)%2, 1});
                double prefactor = 0.5;
                if(useOnlyOneBlock != 0){
                    prefactor = 1.0;
                }
                if(block_to >= 0 and block_from >= 0){
                    Delta[{x, y}] += prefactor*V_sc *(-1.+2*spin) * property_extractor.calculateExpectationValue(
                        {block_from, (spin+1)%2, 1, x,y},
                        {block_to, spin, 0, x,y}
                    );
                }
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
		unsigned int x = from[3];
		unsigned int y = from[4];
		unsigned int spin = from[1];
		unsigned int particleHole = from[2];

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
    if(dimensions == 3){
        Delta = Array<complex<double>>({SIZE_X, SIZE_Y, SIZE_Z});
    }
    else{
        Delta = Array<complex<double>>({SIZE_X, SIZE_Y});
    }
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
        Subindex spin = from[1];
        Subindex particleHole = from[2];
        // int block = getBlockIndex({spin, particleHole});
        // cout << "block: " << block << ", J sign: " << (1. - 2*spin)*(1. - 2*particleHole) << endl;
        return J*(1. - 2*spin)*(1. - 2*particleHole);
    }
    //Set the value for J.
    void setJ(complex<double> J){
        this->J = J;
    }
private:
    complex<double> J;
};

Model generateModel(const complex<double> &mu, const complex<double> &delta, const complex<double> &U, const JCallback &jCallback){
    //Set up the Model.
    Model model;
    for(unsigned int x = 0; x < SIZE_X; x++){
        for(unsigned int y = 0; y < SIZE_Y; y++){
            for(unsigned int spin = 0; spin < 2; spin++){
                for(unsigned int ph = 0; ph < 2; ph++){
                    int block = getBlockIndex({spin, ph});
                    if(block >= 0){
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
                }
                int block_to = getBlockIndex({spin, 0});
                int block_from = getBlockIndex({(spin+1)%2, 1});
                if(block_to >= 0 and block_from >= 0){
                    model << HoppingAmplitude(
                        delta*(1. - 2*spin),
                        {block_to, spin, 0, x, y},
                        {block_from, (spin+1)%2, 1, x, y}
                    ) + HC;
                }
            }
        }
    }
    for(unsigned int spin = 0; spin < 2; spin++){
        for(unsigned int ph = 0; ph < 2; ph++){
            int block = getBlockIndex({spin, ph});
            if(block >= 0){
                model << HoppingAmplitude(
                    jCallback,
                    {block, spin, ph, SIZE_X/2, SIZE_Y/2},
                    {block, spin, ph, SIZE_X/2, SIZE_Y/2}
                );
                model << HoppingAmplitude(
                    U*(1. - 2*ph),
                    {block, spin, ph, SIZE_X/2, SIZE_Y/2},
                    {block, spin, ph, SIZE_X/2, SIZE_Y/2}
                );
            }
        }
    }
    model.construct();
    return model;

}





unique_ptr<Solver::BlockDiagonalizer> solveModel(Model &model){
    unique_ptr<Solver::BlockDiagonalizer> solver(new Solver::BlockDiagonalizer());
    solver->setModel(model);
    solver->setUseGPUAcceleration(USE_GPU);
    solver->setUseMultiGPUAcceleration(USE_MULTI_GPU);
    TBTK::Timer::tick();
    solver->run();
    TBTK::Timer::tock();
    return solver;
}

void runDOScalcs(){
    //as input to the Model.
    JCallback jCallback;
    vector<double> U_list = {0.0, 0.5, -0.5};
    vector<double> J_list = {1.0, 1.0, 1.0};

    Plotter plotter;
    for(unsigned idx = 0; idx < U_list.size(); idx++){
        Model model = generateModel(mu, DELTA, U_list[idx], jCallback);
        //Update the callback with the current value of J.
        jCallback.setJ(J_list[idx]);
        //Set up the Solver.
        unique_ptr<Solver::BlockDiagonalizer> solver = solveModel(model);

        //Set up the PropertyExtractor.
        unsigned resolution = 500;
        PropertyExtractor::BlockDiagonalizer pe(*solver);
        pe.setEnergyWindow(
            -2*DELTA,
            2*DELTA,
            resolution
        );

        TBTK::Index idx_ldos = {IDX_SUM_ALL, IDX_SUM_ALL, 0, SIZE_X/2, SIZE_Y/2};
        
        Property::LDOS ldos = pe.calculateLDOS(
            {idx_ldos}
        );
    

        Array<double> ldos_export({resolution},0);
        Array<double> energies({resolution},0);
        //Store the LDOS in totalLdos.
        for(unsigned int e = 0; e < ldos.getResolution(); e++){
            ldos_export[{e}] = ldos(
                idx_ldos,
                e
            );
            energies[{e}] = ldos.getEnergy(e);
        }


        Exporter exporter;
        exporter.save(ldos_export, DATA_DIR + "ldos_particle_atImp_U_" + to_string(U_list[idx]) + ".csv");
        exporter.save(energies, DATA_DIR + "ldos_particle_atImp_energies_U_" + to_string(U_list[idx]) + ".csv");

        //Smooth the LDOS.
        const double SMOOTHING_SIGMA = 0.01;
        const unsigned int SMOOTHING_WINDOW = 31;
        ldos = Smooth::gaussian(
            ldos,
            SMOOTHING_SIGMA,
            SMOOTHING_WINDOW
        );
    
        plotter.plot(
            idx_ldos,
            ldos
        );
    }

    // plotter.setAxes({
    //     {0, {-5, 5}},
    //     {1, {0, 100}},
    // });
    plotter.setTitle("LDOS at impurity");
    plotter.setLabelX("Energy");
    plotter.setLabelY("LDOS");
    // plotter.setBoundsY(-5, 5);
    plotter.save(FIG_DIR + "ldos_atImp.png");
    //Plot the eigenvalues.
    plotter.clear();
}

int main(){
    //Initialize TBTK.
    Initialize();

    runDOScalcs();
    exit(0);
    //Parameters.
    // const unsigned int SIZE_X = SIZE_X;
    // const unsigned int SIZE_Y = SIZE_Y;
    // const double t = -1;
    // const double mu = -2.;
    // const double Delta = 0.5;
    //Create a callback that returns the Zeeman term and that will be used
    //as input to the Model.
    JCallback jCallback;
    Model model = generateModel(mu, DELTA, U, jCallback);
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
        Solver::BlockDiagonalizer solver;
        solver.setModel(model);
        solver.setUseGPUAcceleration(USE_GPU);
        solver.setUseMultiGPUAcceleration(USE_MULTI_GPU);
        TBTK::Timer::tick();
        solver.run();
        TBTK::Timer::tock();
        //Set up the PropertyExtractor.
        PropertyExtractor::BlockDiagonalizer propertyExtractor(solver);
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
