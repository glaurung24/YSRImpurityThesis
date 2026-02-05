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
#include "TBTK/AnnotatedArray.h"
#include <complex>
#include <memory>
#include "TBTK/Exporter.h"
#include "TBTK/Math/ArrayAlgorithms.h"
using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;
complex<double> i(0, 1);


//Lattice size
const unsigned int SIZE = 21;
const unsigned int SIZE_X = SIZE;
const unsigned int SIZE_Y = SIZE;
const unsigned int SIZE_Z = SIZE;

//Order parameter. 
Array<complex<double>> Delta;

//Superconducting pair potential, convergence limit, max iterations, and initial guess
// and other Parameters.
const complex<double> mu = 0.0;
const complex<double> t = -1.0;
const double V_sc = 1.84; //Started with 3.2
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
const unsigned int DIMENSIONS = 2;

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

vector<Index> generateSpatialIndices()
{
    vector<Index> positionIndexList;
    if (DIMENSIONS == 0)
    {
        return {};
    }
    unsigned num_indices = pow(SIZE, DIMENSIONS);
    positionIndexList.resize(num_indices);
    for (unsigned idx = 0; idx < num_indices; ++idx){
        Index pos;
        unsigned temp = idx;
        for (unsigned dim = 1; dim < DIMENSIONS + 1; ++dim)
        {
            Subindex x = 0;
            unsigned majorityNum = pow(SIZE, DIMENSIONS - dim);
            while (temp >= majorityNum)
            {
                temp -= majorityNum;
                ++x;
            }
            pos.pushBack({x});
        }
        positionIndexList.at(idx) =pos;
    }
    return positionIndexList;
}

Index spatialImpurityIndex()
{
    Index positionImp;
    if (DIMENSIONS == 0)
    {
        return {};
    }
    for (unsigned idx = 0; idx < DIMENSIONS; ++idx){
        positionImp.pushBack(SIZE/2);
    }
    return positionImp;
}

// double getParity(PropertyExtractor::BlockDiagonalizer& pe, unsigned sizeX, unsigned sizeY){
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

bool selfConsistencyStep(const unique_ptr<Solver::BlockDiagonalizer>& solver){ //TODO dimensions (only two atm)
	PropertyExtractor::BlockDiagonalizer property_extractor(*solver);
	Array<complex<double>> delta_old = Delta;
    if(DIMENSIONS != 2){
        cerr << "only 2D supported in selfConsistencyStep()" << endl;
        exit(-1);
    }
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
        unsigned z = -1;
        if(DIMENSIONS == 3){
            z = from[5];
        }
		unsigned int spin = from[1];
		unsigned int particleHole = from[2];

        if(DIMENSIONS == 2){
            if(spin == 0 && particleHole == 0)
                return conj(Delta[{x, y}]);
            else if(spin == 1 && particleHole == 0)
                return -conj(Delta[{x, y}]);
            else if(spin == 0 && particleHole == 1)
                return -Delta[{x, y}];
            else
                return Delta[{x, y}];
        }
        else{
            if(spin == 0 && particleHole == 0)
                return conj(Delta[{x, y, z}]);
            else if(spin == 1 && particleHole == 0)
                return -conj(Delta[{x, y, z}]);
            else if(spin == 0 && particleHole == 1)
                return -Delta[{x, y, z}];
            else
                return Delta[{x, y, z}];
        }

	}
} deltaCallback;

//Function responsible for initializing the order parameter
void initDelta(){
    if(DIMENSIONS == 3){
        Delta = Array<complex<double>>({SIZE_X, SIZE_Y, SIZE_Z}, DELTA);
    }
    else{
        Delta = Array<complex<double>>({SIZE_X, SIZE_Y}, DELTA);
    }
	// const double rand_spread = DELTA_INITIAL_GUESS_RANDOM_WINDOW*abs(DELTA_INITIAL_GUESS);
	// srand (static_cast <unsigned> (time(0)));
	// for(unsigned int x = 0; x < SIZE_X; x++){
	// 	for(unsigned int y = 0; y < SIZE_Y; y++){
	// 		double a = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
	// 		double b = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
	// 		Delta[{x, y}] = DELTA_INITIAL_GUESS + rand_spread*(a+i*b);
	// 	}
	// }
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

Model generateModel(const complex<double> &mu, const complex<double> &delta, const complex<double> &U, const JCallback &jCallback, const DeltaCallback &deltaCallback){
    //Set up the Model.
    Model model;
    vector<Index> spatialPositions = generateSpatialIndices();
    for(auto onsite_pos : spatialPositions){
        for(unsigned int spin = 0; spin < 2; spin++){
            for(unsigned int ph = 0; ph < 2; ph++){
                int block = getBlockIndex({spin, ph});
                Index onsiteIndex({block, spin, ph});
                for (unsigned index = 0; index < onsite_pos.getSize(); ++index)
                {
                    onsiteIndex.pushBack(onsite_pos.at(index));
                }
                if(block >= 0){
                    model << HoppingAmplitude(
                        -mu*(1. - 2*ph),
                        {onsiteIndex},
                        {onsiteIndex}
                    );
                    for (unsigned dim = 0; dim < DIMENSIONS; ++dim)
                    {
                        Index hoppingIndex({block, spin, ph});
                        for (unsigned index = 0; index < onsite_pos.getSize(); ++index)
                        {
                            if (index == dim)
                            { // Hopping for this dimension
                                hoppingIndex.pushBack((onsite_pos.at(index) + 1) % SIZE);
                            }
                            else
                            {
                                hoppingIndex.pushBack(onsite_pos.at(index));
                            }
                        }
                        model << HoppingAmplitude(
                                    -t * (1. - 2 * ph),
                                    hoppingIndex,
                                    onsiteIndex) +
                                    HC;
                    }
                }
            }
            int block_to = getBlockIndex({spin, 0});
            int block_from = getBlockIndex({(spin+1)%2, 1});
            Index onsiteIndex_to({block_to, spin, 0});
            Index onsiteIndex_from({block_from, (spin+1)%2, 1});
            for (unsigned index = 0; index < onsite_pos.getSize(); ++index)
            {
                onsiteIndex_to.pushBack(onsite_pos.at(index));
                onsiteIndex_from.pushBack(onsite_pos.at(index));
            }
            if(block_to >= 0 and block_from >= 0){
                model << HoppingAmplitude(
                    deltaCallback,
                    onsiteIndex_to,
                    onsiteIndex_from
                ) + HC;
            }
        }
    }
    Index impIndex = spatialImpurityIndex();
    for(unsigned int spin = 0; spin < 2; spin++){
        for(unsigned int ph = 0; ph < 2; ph++){
            int block = getBlockIndex({spin, ph});
            Index onsiteIndex = {block, spin, ph};
            for (unsigned index = 0; index < impIndex.getSize(); ++index){
                onsiteIndex.pushBack(impIndex.at(index));
            }
            if(block >= 0){
                model << HoppingAmplitude(
                    jCallback,
                    onsiteIndex,
                    onsiteIndex
                );
                model << HoppingAmplitude(
                    U*(1. - 2*ph),
                    onsiteIndex,
                    onsiteIndex
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
        DeltaCallback deltaCallback;
        initDelta();
        Model model = generateModel(mu, DELTA, U_list[idx], jCallback, deltaCallback);
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

        Index impuritySite = spatialImpurityIndex();
        TBTK::Index idx_ldos = {IDX_SUM_ALL, IDX_SUM_ALL, 0};
        for (unsigned index = 0; index < impuritySite.getSize(); ++index){
            idx_ldos.pushBack(impuritySite.at(index));
        }
    
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

vector<complex<double>> calculateExpectationValuesPerState(const unique_ptr<PropertyExtractor::BlockDiagonalizer>& pe, const Index& to, const Index& from,
                                                            const vector<Subindex>& stateIndices){
    int nr_states = stateIndices.size();
    vector<complex<double>> values(nr_states, 0.);
    vector<Index> to_patterns(nr_states, to);
    vector<Index> from_patterns(nr_states, from);
    Property::WaveFunctions wf_to = pe->calculateWaveFunctions(to_patterns, stateIndices);
    Property::WaveFunctions wf_from = pe->calculateWaveFunctions(from_patterns, stateIndices);
    
	for(Subindex n = 0; n < nr_states; n++){
        complex<double> u_to = wf_to(to_patterns[n], stateIndices[n]);
        complex<double> u_from = wf_from(from_patterns[n], stateIndices[n]);

        values[n] += conj(u_to)*u_from; //TODO warning this does not seem to work as intendet with IDX sum all
        cout << values[n] << endl;
    }
    return values;
}

vector<double> calculateMagnetizationPerState(const unique_ptr<PropertyExtractor::BlockDiagonalizer>& pe, const vector<Subindex>& stateIndices){
    int nr_states = stateIndices.size();
    vector<double> magnetizations(nr_states,0.);
    Index pattern_all = {IDX_ALL, IDX_ALL, IDX_ALL};
    for(unsigned i = 0; i < DIMENSIONS; i++){
        pattern_all.pushBack(IDX_ALL);
    }
    vector<Index> patterns_all(nr_states, pattern_all);
    Property::WaveFunctions wavefcts = pe->calculateWaveFunctions(patterns_all, stateIndices);
    vector<Index> spatialPositions = generateSpatialIndices();
    #pragma omp for
    for(auto onsite_pos : spatialPositions){
        for(unsigned int spin = 0; spin < 2; spin++){
            double prefactor = (1. - 2*spin);
            for(unsigned int ph = 0; ph < 2; ph++){
                prefactor *= (1. - 2*ph);
                int block = getBlockIndex({spin, ph});
                if(block >= 0){
                    Index pattern = {block}; //block
                    pattern.pushBack(spin);
                    pattern.pushBack(ph);
                    for(unsigned dim = 0; dim < DIMENSIONS; dim++){
                        pattern.pushBack(onsite_pos.at(dim));
                    }
                    for(unsigned idx = 0; idx <  stateIndices.size(); idx++){
                        complex<double> wf = wavefcts(pattern, stateIndices[idx]);
                        magnetizations[idx] += prefactor*real(conj(wf)*wf);
                    }
                }

            }
        }
    }
    return magnetizations;
}

double calculateTotalMagnetization(const unique_ptr<PropertyExtractor::BlockDiagonalizer>& pe){
    double magnetization_expectation_value = 0.;
    vector<Index> onsitePostions = generateSpatialIndices();
    #pragma omp for
    for(auto onsite_pos : onsitePostions){
        for(unsigned int spin = 0; spin < 2; spin++){
            for(unsigned int ph = 0; ph < 2; ph++){
                int block = getBlockIndex({spin, ph});
                if(block >= 0){
                    Index onsiteIndex = {block};
                    onsiteIndex.pushBack(spin);
                    onsiteIndex.pushBack(ph); // ph
                    for (unsigned index = 0; index < onsite_pos.getSize(); ++index)
                    {
                        onsiteIndex.pushBack(onsite_pos.at(index));
                    }
                    magnetization_expectation_value += (1. - 2*spin)*(1. - 2*ph)*
                                                 real(pe->calculateExpectationValue(onsiteIndex, onsiteIndex));
                }
            }
        }
    }
    return 0.5*magnetization_expectation_value;
}

vector<complex<double>> calculatePairingPerStateAtImpSite(const unique_ptr<PropertyExtractor::BlockDiagonalizer>& pe, const vector<Subindex>& stateIndices){
    int nr_states = stateIndices.size();
    vector<complex<double>> pairings(nr_states,0.);
    Index pattern_all = {IDX_ALL, IDX_ALL, IDX_ALL};
    for(unsigned i = 0; i < DIMENSIONS; i++){
        pattern_all.pushBack(IDX_ALL);
    }
    vector<Index> patterns_all(nr_states, pattern_all);
    Property::WaveFunctions wavefcts = pe->calculateWaveFunctions(patterns_all, stateIndices);
    // vector<Index> spatialPositions = generateSpatialIndices();
    // for(auto onsite_pos : spatialPositions){
    Index impSite = spatialImpurityIndex();
        for(unsigned spin = 0; spin < 2; ++spin){
            // Gap equation
            int block_to = getBlockIndex({spin, 0});
            int block_from = getBlockIndex({(spin+1)%2, 1});
            double prefactor = 0.5;
            if(useOnlyOneBlock != 0){
                prefactor = 1.0;
            }
            Index pattern_from = {block_from, (spin+1)%2, 1};
            Index pattern_to = {block_to, spin, 0};
            for (unsigned index = 0; index < impSite.getSize(); ++index)
            {
                pattern_from.pushBack(impSite.at(index));
                pattern_to.pushBack(impSite.at(index));
            }
            if(block_to >= 0 and block_from >= 0){
                for(unsigned idx = 0; idx <  stateIndices.size(); idx++){
                    complex<double> wf_from = wavefcts(pattern_from, stateIndices[idx]);
                    complex<double> wf_to = wavefcts(pattern_to, stateIndices[idx]);
                    pairings[idx] += prefactor *(-1.+2*spin) * conj(wf_from)*wf_to;
                }
            }
        }
    // }
    return pairings;
}

void energySpectrum(){

    JCallback jCallback;
    DeltaCallback deltaCallback;
    initDelta();
    Model model = generateModel(mu, DELTA, U, jCallback, deltaCallback);


    //Number of iterations.
    const unsigned int NUM_ITERATIONS = 50;
    //Arrays where the results are stored after each iteration.
    Array<double> totalLdos({NUM_ITERATIONS, 500}, 0);
    Array<double> totalLdosSmooth({NUM_ITERATIONS, 500}, 0);
    Array<double> totalEigenValues({
        NUM_ITERATIONS,
        (unsigned int)model.getBasisSize()
    });
    //Iterate over 100 values for J.
    Range j(0, 5, NUM_ITERATIONS);

    Array<double> j_values(j);
    Exporter exporter;
    exporter.save(j_values, DATA_DIR + "jvalues.csv");

    for(unsigned int n = 0; n < NUM_ITERATIONS; n++){
        //Update the callback with the current value of J.
        jCallback.setJ(j[n]);
        //Set up the Solver.
        unique_ptr<Solver::BlockDiagonalizer> solver = solveModel(model);
        //Set up the PropertyExtractor.
        PropertyExtractor::BlockDiagonalizer propertyExtractor(*solver);

		std::cout << "J: " << j[n] << std::endl;
		// getParity(propertyExtractor, SIZE_X, SIZE_Y) ;
        //Calculate the eigenvalues.
        Property::EigenValues eigenValues
            = propertyExtractor.getEigenValues();

        const double LOWER_BOUND = -2*DELTA;
        const double UPPER_BOUND = 2*DELTA;
        vector<Subindex> energyIndices;
        //Calculate Magnetization per eigenvalue within the bounds
        for(unsigned idx_ev = 0; idx_ev < eigenValues.getSize(); idx_ev++){
            // if(eigenValues(idx_ev) > LOWER_BOUND and eigenValues(idx_ev) < UPPER_BOUND){
                energyIndices.push_back(idx_ev);
            // }
        }
        vector<double> magnetizations = calculateMagnetizationPerState(make_unique<PropertyExtractor::BlockDiagonalizer>(propertyExtractor), energyIndices);

        double totalMag = 0.;
        for(unsigned idx = 0; idx < eigenValues.getSize(); idx++){
            
            if(eigenValues(idx) < 0.){
                totalMag += magnetizations[idx];
            }
        }
        cout << totalMag << endl;

        // cout << "Total Magnetization: " << calculateTotalMagnetization(make_unique<PropertyExtractor::BlockDiagonalizer>(propertyExtractor)) << endl;

        exporter.save(Array<double>(magnetizations), DATA_DIR + "magnetizations_per_state_J_" + to_string(j[n]) + ".csv");
        exporter.save(eigenValues, DATA_DIR + "eigenvalues_J_" + to_string(j[n]) + ".csv");

        vector<complex<double>> pairings = calculatePairingPerStateAtImpSite(make_unique<PropertyExtractor::BlockDiagonalizer>(propertyExtractor), energyIndices);
        exporter.save(Math::ArrayAlgorithms<complex<double>>::real(Array<complex<double>>(pairings)), DATA_DIR + "pairings_per_state_real_J_" + to_string(j[n]) + ".csv");
        exporter.save(Math::ArrayAlgorithms<complex<double>>::imag(Array<complex<double>>(pairings)), DATA_DIR + "pairings_per_state_imag_J_" + to_string(j[n]) + ".csv");

        double deltaAtImp = 0.;
        for(unsigned idx = 0; idx < eigenValues.getSize(); idx++){
            
            if(eigenValues(idx) < 0.){
                deltaAtImp += V_sc* real(pairings[idx]);
            }
        }
        cout << "Delta at imp:" << deltaAtImp << endl;
        // Calculate the local density of states (LDOS).
        Index impuritySite = spatialImpurityIndex();
        TBTK::Index idx_ldos = {IDX_SUM_ALL, IDX_SUM_ALL, IDX_SUM_ALL};
        for (unsigned index = 0; index < impuritySite.getSize(); ++index){
            idx_ldos.pushBack(impuritySite.at(index));
        }
        const unsigned int RESOLUTION = 500;
        propertyExtractor.setEnergyWindow(
            LOWER_BOUND,
            UPPER_BOUND,
            RESOLUTION
        );
        Property::LDOS ldos = propertyExtractor.calculateLDOS({
            idx_ldos
        });

        Array<double> ldos_export({RESOLUTION},0);
        //Save the ldos
        for(unsigned int e = 0; e < ldos.getResolution(); e++){
            ldos_export[{e}] = ldos(
                idx_ldos,
                e
            );
        }
        exporter.save(ldos_export, DATA_DIR + "ldos_J_"+ to_string(j[n]) + ".csv");
        //Smooth the LDOS.
        const double SMOOTHING_SIGMA = 0.01;
        const unsigned int SMOOTHING_WINDOW = 31;
        Property::LDOS ldosSmooth = Smooth::gaussian(
            ldos,
            SMOOTHING_SIGMA,
            SMOOTHING_WINDOW
        );
        //Store the LDOS in totalLdos.
        for(unsigned int e = 0; e < ldos.getResolution(); e++){
            totalLdos[{n, e}] = ldos(
                idx_ldos,
                e
            );
            totalLdosSmooth[{n, e}] = ldosSmooth(
                idx_ldos,
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
        {1, {-2*DELTA, 2*DELTA}},
    });
    plotter.setTitle("LDOS");
    plotter.setLabelX("J");
    plotter.setLabelY("Energy");
    plotter.setBoundsY(-2*abs(DELTA), 2*abs(DELTA));
    plotter.plot(totalLdosSmooth);
    plotter.save(FIG_DIR + "LDOS.png");
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
    plotter.save(FIG_DIR + "EigenValues.png");
}



vector<Subindex> getYSTStateIndex(const unique_ptr<PropertyExtractor::BlockDiagonalizer>& pe){
    Property::EigenValues ev = pe->getEigenValues();
    Subindex lowerIndex = -1;
    Subindex upperIndex = -1;
    double positiveYSREnergy = 4.; //upper band limit
    double negativeYSREnergy = -4.;
    for(unsigned index = 0; index < ev.getSize(); index++){
        if(ev(index) > 0. && ev(index) < positiveYSREnergy){
            positiveYSREnergy = ev(index);
            upperIndex = index;
        }
        if(ev(index) < 0. && ev(index) > negativeYSREnergy){
            negativeYSREnergy = ev(index);
            lowerIndex = index;
        }
    }
    return vector<Subindex>({lowerIndex, upperIndex});
}

Array<double> calculateLocalizationYSR(const unique_ptr<PropertyExtractor::BlockDiagonalizer>& pe, unsigned radius){
    vector<Subindex> ysrIndices = getYSTStateIndex(pe);
    Subindex YSRstate = ysrIndices[0];
    Array<double> localization({radius*2,radius*2}, 0.);
    if(DIMENSIONS != 2){
        cout << "calculateLocalizationYSR() in line " << __LINE__ << " only supports 2D calculations (for now)" << endl;
        return localization;
    }

    Index pattern_all = {IDX_ALL, IDX_ALL, IDX_ALL};
    for(unsigned i = 0; i < DIMENSIONS; i++){
        pattern_all.pushBack(IDX_ALL);
    }
    Property::WaveFunctions wavefcts = pe->calculateWaveFunctions({pattern_all}, {YSRstate});
    vector<Index> spatialPositions = generateSpatialIndices();
    #pragma omp for
    for(unsigned x = SIZE_X/2-radius; x < SIZE/2+radius; x++){
        for(unsigned y = SIZE_Y/2-radius; y < SIZE/2+radius; y++){
            for(unsigned int spin = 0; spin < 2; spin++){
                for(unsigned int ph = 0; ph < 2; ph++){
                    int block = getBlockIndex({spin, ph});
                    if(block >= 0){
                        Index pattern = {block}; //block
                        pattern.pushBack(spin);
                        pattern.pushBack(ph);
                        pattern.pushBack(x);
                        pattern.pushBack(y);
                        // for(unsigned dim = 0; dim < DIMENSIONS; dim++){
                        //     pattern.pushBack(onsite_pos.at(dim));
                        // }
                        complex<double> wf = wavefcts(pattern, YSRstate);
                        unsigned x_index = x + radius - SIZE_X/2;
                        unsigned y_index = y + radius - SIZE_Y/2;
                        // cout << x_index << ", " << y_index << endl;
                        // if(x_index == 0 && y_index == 0 ){
                        //     cout << real(conj(wf)*wf) << endl;
                        // }
                        localization[{{x_index},{y_index}}] += real(conj(wf)*wf);
                    }

                }
            }
        }
    }
    return localization;
}

void localization(){
    JCallback jCallback;
    unsigned radius = 10;
    vector<complex<double>> mu_list = {0.0, 2., 3.618};
    Exporter exporter;
    for(auto mu : mu_list){
        DeltaCallback deltaCallback;
        initDelta();
        Model model = generateModel(mu, DELTA, U, jCallback, deltaCallback);
        //Update the callback with the current value of J.
        jCallback.setJ(J);
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
        Array<double> localization = calculateLocalizationYSR(make_unique<PropertyExtractor::BlockDiagonalizer>(pe), radius);
        vector<double> localization_vec;
        for(unsigned x = 0; x < 2*radius; x++){
            for(unsigned y = 0; y < 2*radius; y++){
                localization_vec.push_back(localization[{{x},{y}}]);
            }
        }
        exporter.save(Array<double>(localization_vec), DATA_DIR + "localization_mu_" + to_string(real(mu)) + ".csv");
    }


}

void selfConsistency(){
    JCallback jCallback;
    DeltaCallback deltaCallback;
    initDelta();
    Model model = generateModel(mu, DELTA, U, jCallback, deltaCallback);


    //Number of iterations.
    const unsigned int NUM_ITERATIONS = 30;
    const unsigned int MAX_SC_ITERATIONS = 100;
    //Arrays where the results are stored after each iteration.
    Array<double> totalLdos({NUM_ITERATIONS, 500}, 0);
    Array<double> totalLdosSmooth({NUM_ITERATIONS, 500}, 0);
    Array<double> totalEigenValues({
        NUM_ITERATIONS,
        (unsigned int)model.getBasisSize()
    });
    //Iterate over 100 values for J.
    Range j(0, 5, NUM_ITERATIONS);

    Array<double> j_values(j);
    Exporter exporter;
    exporter.save(j_values, DATA_DIR + "jvalues_sc.csv");

    for(unsigned int n = 0; n < NUM_ITERATIONS; n++){
        //Update the callback with the current value of J.
        jCallback.setJ(j[n]);
        //Set up the Solver.
        unique_ptr<Solver::BlockDiagonalizer> solver = solveModel(model);
        //Set up the PropertyExtractor.
        unsigned iteration = 0;
        for(; iteration <= MAX_SC_ITERATIONS; iteration++){
            if(selfConsistencyStep(solver)){
                break;
            }
            solver->run();
            cout << "Sc iteration nr: " << iteration << endl;
            cout << Delta[{SIZE/2, SIZE/2}] << endl;
        }
        if(iteration >= MAX_SC_ITERATIONS){
            cerr << "Maximum sc iterations reached" << endl;
        }



        PropertyExtractor::BlockDiagonalizer propertyExtractor(*solver);

		std::cout << "J: " << j[n] << std::endl;

        Property::EigenValues ev = propertyExtractor.getEigenValues();
        vector<Subindex> energyIndices;
        //Calculate Magnetization per eigenvalue within the bounds
        for(unsigned idx_ev = 0; idx_ev < ev.getSize(); idx_ev++){
            // if(eigenValues(idx_ev) > LOWER_BOUND and eigenValues(idx_ev) < UPPER_BOUND){
                energyIndices.push_back(idx_ev);
            // }
        }

        vector<double> delta_vec;
        for(unsigned x = 0; x < SIZE; x++){
            for(unsigned y = 0; y < SIZE; y++){
                delta_vec.push_back(real(Delta[{{x},{y}}]));
            }
        }
        exporter.save(Array<double>(delta_vec), DATA_DIR + "delta_sc_J_" + to_string(j[n]) + ".csv");
        exporter.save(ev, DATA_DIR + "eigenvalues_sc_J_" + to_string(j[n]) + ".csv");

        vector<complex<double>> pairings = calculatePairingPerStateAtImpSite(make_unique<PropertyExtractor::BlockDiagonalizer>(propertyExtractor), energyIndices);
        exporter.save(Math::ArrayAlgorithms<complex<double>>::real(Array<complex<double>>(pairings)), DATA_DIR + "pairings_per_state_sc_real_J_" + to_string(j[n]) + ".csv");
        exporter.save(Math::ArrayAlgorithms<complex<double>>::imag(Array<complex<double>>(pairings)), DATA_DIR + "pairings_per_state_sc_imag_J_" + to_string(j[n]) + ".csv");


    }
    vector<complex<double>> mu_list = {0.0, 2., 3.618};
    for(auto mu : mu_list){
        DeltaCallback deltaCallback;
        initDelta();
        Model model = generateModel(mu, DELTA, U, jCallback, deltaCallback);
        //Update the callback with the current value of J.
        jCallback.setJ(J);
        //Set up the Solver.
        unique_ptr<Solver::BlockDiagonalizer> solver = solveModel(model);
        //Set up the PropertyExtractor.
        unsigned iteration = 0;
        for(; iteration <= MAX_SC_ITERATIONS; iteration++){
            if(selfConsistencyStep(solver)){
                break;
            }
            solver->run();
            cout << "Sc iteration nr: " << iteration << endl;
            cout << Delta[{SIZE/2, SIZE/2}] << endl;
        }
        if(iteration >= MAX_SC_ITERATIONS){
            cerr << "Maximum sc iterations reached" << endl;
        }
        vector<double> delta_vec;
        for(unsigned x = 0; x < SIZE; x++){
            for(unsigned y = 0; y < SIZE; y++){
                delta_vec.push_back(real(Delta[{{x},{y}}]));
            }
        }
        exporter.save(Array<double>(delta_vec), DATA_DIR + "delta_sc_mu_" + to_string(real(mu)) + ".csv");

    }

}

int main(){
    //Initialize TBTK.
    Initialize();

    // runDOScalcs();
    // energySpectrum();
    // localization();
    selfConsistency();
    return 0;
}
