package edu.boun.edgecloudsim.applications.sample_app5;
import edu.boun.edgecloudsim.utils.SimLogger;


import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;

import weka.core.CapabilitiesHandler;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Standardize;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

// Writing to file
import java.io.File;  // Import the File class
import java.io.FileWriter;
import java.io.IOException;

import java.util.Arrays;
import java.util.Collections;
import java.util.ArrayList;
import java.util.Random;



public class DataWrapper {
	
	// ******************* Global variables for this class only *******************
	private static int num_of_rows = 200;  // the number of samples/rows to collect from SimLogger ******
	private static String nom_de_classes = "binary";  //"categorical"
	private static double EPSILON = 0.5;
	private static double ALPHA = 1e-3; 	//learning rate  
	private static Instances state_train;
	private static MultilayerPerceptron q_network;
	private static MultilayerPerceptron target_q_network;
	private static Instances state_instances;
	private static Instance state_instance;
	private static double action_greedy;
	private static double R;
	private static double done;
	
	private static double GAMMA = 0.995; //discount factor
	private static int update_interval = 10; //count frequency of updating the target_q_network
	private static double NU = 0.1;
	
	// these are all size update_interval
	private static Instances [][] s_t;
	private static double [] a_t;
	private static double [] R_t;
	private static double [] y_t;
	private static double [] d_t;
	// ******************* 
	
	
	// Where do these feature names come from?? They all come from SimLogger.java
	private static final String[] STATE_DATA_ATTRIBUTES = {"Tall","Cj_rem","processing_time", "CPUcost", "BWcost", "packet_loss_rate", "getNetworkDelay", "result_of_task", "QOE", "y_actual_binary"};

	
	private static DataWrapper singleton = new DataWrapper();

	/*
	 * A private Constructor prevents any other class from instantiating.
	 */
	DataWrapper() {

	}

	/* Static 'instance' method */
	public static DataWrapper getInstance() {
		return singleton;
	}
	
	
	public void train_models(String wekaModelsFolder) throws Exception {
		// -----------------------------------------------
		// Initialize experience playback matrices size
		a_t = new double[update_interval];  // double à chaque instance
		s_t = new Instances[update_interval][1];  //vector à chaque instance, using state_instance instead
		R_t = new double[update_interval];
		y_t = new double[update_interval];
		d_t = new double[update_interval];
		// -----------------------------------------------
		
		// -----------------------------------------------
		// [Step 1] Gather input data
		System.out.println("[Step 1] Gather input data\n");
		state_train = read_data_from_file();
		// -----------------------------------------------
		
		// -----------------------------------------------
		// [Step 2] Train model: Defines the model as 'MultilayerPerceptron'
		System.out.println("[Step 2] Train model\n");
		q_network = train_qlearning_nns(state_train);
		target_q_network = train_qlearning_nns(state_train);
		// -----------------------------------------------
		
		// -----------------------------------------------
		// [Step 3] Verify that initial trained model is decent
		System.out.println("[Step 3] Evaluate q_network model\n");
		evaluate_model(q_network);
		
		System.out.println("[Step 3] Evaluate target_q_network model\n");
		evaluate_model(target_q_network);
		// -----------------------------------------------
		
		// -----------------------------------------------
		// Save the q_network and target_q_network models
		String Savefile_PATH = wekaModelsFolder + "q_network.model";
		String Savefile_PATH2 = wekaModelsFolder + "target_q_network.model";
		try {
			weka.core.SerializationHelper.write(Savefile_PATH, q_network);
			weka.core.SerializationHelper.write(Savefile_PATH2, target_q_network);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// -----------------------------------------------
	}
	
	
	
	public void initialize(String ClassifierType, String wekaModelsFolder) {
		
		// -----------------------------------------------
		// Initialize experience playback matrices size
		a_t = new double[update_interval];  // double à chaque instance
		s_t = new Instances[update_interval][1];  //vector à chaque instance, using state_instance instead
		R_t = new double[update_interval];
		y_t = new double[update_interval];
		d_t = new double[update_interval];
		// -----------------------------------------------
		
		try {
			if(ClassifierType.equals("MultilayerPerceptron")) {
				q_network = (MultilayerPerceptron) weka.core.SerializationHelper.read(wekaModelsFolder + "q_network.model");
				target_q_network = (MultilayerPerceptron) weka.core.SerializationHelper.read(wekaModelsFolder + "target_q_network.model");
			}
		}
		catch (Exception e) {
			SimLogger.printLine("cannot serialize weka objects!");
			System.exit(1);
		}
	}
	
	
	
	public double [] main() throws Exception {
		
		// Initialize result (return of function)
		Random rand = new Random();
		int result = rand.nextInt(2)+1;
		
		try{
			// -----------------------------------------------
			System.out.println("[Step 5] Calculate action/command to network\n");
			action_greedy = predict_action_AND_rewards();
			// -----------------------------------------------
			
			// -----------------------------------------------
			System.out.println("[Step 6] Measure network result based on action/command\n");
			result = execute_action();
			// -----------------------------------------------
			
			// -----------------------------------------------
			System.out.println("[Step 7] Calculate reward\n");
			R = calculate_reward(); // 
			// -----------------------------------------------
			
			// -----------------------------------------------
			System.out.println("[Step 8] Determine if episode is done (measure resources remaining on EDGE) [0=not done, 1=done]\n");
			done = determine_if_episode_done();
			
			// -----------------------------------------------
			
			// -----------------------------------------------
			System.out.println("[Step 9] Store past values\n");
			store_qlearning_values();
			
			// -----------------------------------------------
			
			// -----------------------------------------------
			System.out.println("[Step 10] Calculate loss, update weight on gradients for backpropagation\n");
			agent_learn(done);
			// -----------------------------------------------
			
		}
		catch(Exception ex){
			ex.printStackTrace();
		}  // end of try
		
		double [] output = {result, done, R};
		
		return output;
	} // End of main_train()
	
	
	
	
	
			
	
	
	
	
	
	// ----------------------------------------------
	// Training initial models
	// ----------------------------------------------
	public static Instances read_data_from_file() throws Exception {
		
		//String CSV_FilePath = "C:\\Users\\jafouche\\Documents\\Edge_computing\\XNETmatrix.txt";
		String CSV_FilePath = "C:\\Users\\jafouche\\Documents\\Edge_computing\\EdgeCloudSim\\scripts\\sample_app5\\config\\weka\\XNETmatrix.txt";
		CSVLoader loader = new CSVLoader();
		String[] options = new String[5];
		options[0] = "-H";
		options[1] = "-F";
		options[2] = ",";
		options[3] = "-R";
		options[4] = "1,2,3,4,5,6,7,8,9,10-last";
		loader.setOptions(options);
		loader.setSource(new File(CSV_FilePath));
		loader.setNoHeaderRowPresent(true);
		Instances state_train = loader.getDataSet();
		
		// Set the class
		System.out.println("Number of columns=state_train.numAttributes(): " + state_train.numAttributes());
		state_train.setClassIndex(state_train.numAttributes() - 1);
		
		// -----------------------
		// setup a filter to transform the numerical values in the last column to nominal/'string' 
		// smote can only use non-numeric class values
		NumericToNominal filter1 = new NumericToNominal(); // Converting last index attribute to type
		filter1.setAttributeIndices("last"); // as the last index would be class
		filter1.setInputFormat(state_train);  // specify the input instance format for the filter
		state_train = Filter.useFilter(state_train, filter1); // filtered
		
		// Balance the classes 
		SMOTE smote=new SMOTE();
	    smote.setInputFormat(state_train);       
	    state_train = Filter.useFilter(state_train, smote);
	    // -----------------------
		
		// Print header and instances: output of zero
		//System.out.println("state_train\n");
		//System.out.println(state_train);  // Outputs data, attribute, relation
		
		return state_train;
	}
	
	
	public static MultilayerPerceptron train_qlearning_nns(Instances state_train) throws Exception {
		MultilayerPerceptron q_network = new MultilayerPerceptron();
		//Setting Parameters
		q_network.setLearningRate(ALPHA);
		q_network.setMomentum(0.2);
		q_network.setTrainingTime(100); // number of training epochs
		q_network.setHiddenLayers("10,5,2"); // make a 2-layer NN, with binary output
	    
		q_network.buildClassifier(state_train);
		
		return q_network;
	}


	public static void evaluate_model(MultilayerPerceptron model) throws Exception {
		
		Evaluation evaluation = new Evaluation(state_train);
		evaluation.evaluateModel(model, state_train); // the output is an object of various evaluation metrics
		//System.out.println(evaluation.toSummaryString());
		
		// For Initially trained model we just need to know the accuracy- (prediction evaluations across all samples of train data)
		// We do not need to output the prediction evaluations
		double accuracy = evaluation.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
		
		// Best way to predict new instances
	    state_instance = state_train.firstInstance();  // Take one instance, WORKS!!
	    
	    double[] prediction_prob = model.distributionForInstance(state_instance);
	    System.out.println("prediction_prob.length = " + prediction_prob.length);
	    System.out.println("prediction_prob = ");
	    for (int i=0;i<prediction_prob.length;i++){
	        System.out.println(prediction_prob[i] + " ");
	    }
	    
	    // Test classifyInstance
	    double result = model.classifyInstance(state_instance);
	    System.out.println("classifyInstance = ");
	    System.out.println(result);
	}
	// ----------------------------------------------
	
	
	// ----------------------------------------------
	// Q-learning algorithm
	// ----------------------------------------------
	private double predict_action_AND_rewards() {
		
		// -----------------------------------
		// Call for data samples
		state_instances = data_input_logger(num_of_rows, "current_state", STATE_DATA_ATTRIBUTES);
		
		// Convert data sample Instances to Instance
		state_instance = state_instances.firstInstance();  // Take one instance, WORKS!!
		// OU
		// Instance state_instance = state_instances.instance(0);  // Take one instance, WORKS!!
		// -----------------------------------
		
		// -----------------------------------
		// If EPSILON is small it does exploitation, and if EPSILON is big it does exploration
		Random rand = new Random();
		double p = rand.nextDouble();
		System.out.println("check for p from 0 to 1 : " + p);
		
		// Initialize and case where (p < EPSILON)
		// Exploration : randomly select an action
		double action_greedy = rand.nextInt(1); //for y_actual_binary = [0="EDGE", 1="NOEDGE"]
		//double action_greedy = rand.nextInt(2); //for y_actual_binary = [0=EDGE(VmType=1), 1=CLOUDGSM(VmType=2), 2=CLOUDRSU(VmType=2)]
		
		
		if (p > EPSILON) {
			System.out.println("Exploitation: Agent selected based on q_network prediction.");
			// Predict using current q_network
			try {
				// Returns a non-class value sometimes instead of the classes
				//action_greedy = q_network.classifyInstance(state_instance);  // Output results of model fit [0="EDGE", 1="NOEDGE"]
				//OU
				action_greedy = predict(state_instance, q_network);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		else {
			System.out.println("Exploration: Agent selected based randomely.");
		}
		
		System.out.println("action_greedy ([0=EDGE, 1=NOEDGE]), meaning command the network to select this type of communication : ");
		System.out.println("action_greedy : " + action_greedy);
		
		return action_greedy;
	}
	
	
	private int execute_action() {
		int result;
		Random rand = new Random();
		
		// Command the network with respect to the action
		if (nom_de_classes == "binary")
		{
			// y_actual_binary
			if (action_greedy == 0){
				result = VehicularEdgeOrchestrator.EDGE_DATACENTER;
				// result = 3
			}
			else {
				// Randomly select between GSM and RSU
				if (rand.nextInt(1)+1 == 1) { 
					result = VehicularEdgeOrchestrator.CLOUD_DATACENTER_VIA_GSM;
					//result = 1
				}
				else {
					result = VehicularEdgeOrchestrator.CLOUD_DATACENTER_VIA_RSU;
					//result = 2
				}
			}
		}
		else { //(nom_de_classes == "categorical")
			if (action_greedy == 0){
				result = VehicularEdgeOrchestrator.EDGE_DATACENTER;
				//result = 3
			}
			else if (action_greedy == 1){
				result = VehicularEdgeOrchestrator.CLOUD_DATACENTER_VIA_GSM;
				//result = 1
			}
			else {//if (action_greedy == 2){
				result = VehicularEdgeOrchestrator.CLOUD_DATACENTER_VIA_RSU;
				// result = 2
			}
		}
		
		System.out.println("result ([1=GSM, 2=RSU, 3=EDGE]), meaning the following network as selected : ");
		System.out.println("result : " + result);
		
		return result;
	}
	
	
	
	public double calculate_reward() {
		// Calculate the reward for the action taken, based on the measured result of task.
		double RoT = SimLogger.result_of_task; //result_of_task = [0=success, 1=fail, 2=timeout]
		double R;
		
		// Calculate the reward R
		if (RoT == 1) {
			// fail
			R = 0;
		}
		else if (RoT == 0) {
			// accomplished
			R = SimLogger.QOE;
		}
		else {
			// (RoT == 2) timeout
			R = -1;
		}
		
		System.out.println("result_of_task ([0=success, 1=fail, 2=timeout]), after commanding the network we measure the network to understand the network's response based on our action :");
		System.out.println("result_of_task : " + RoT);
		System.out.println("reward R, meaning the agent received this reward for their selection : ");
		System.out.println("R : " + R);
		
		return R;
	}
	
	
	public double determine_if_episode_done() {
		// check if episode if finished
		double remaining_resources = SimLogger.Cj_rem; // made it a variable for clarity that Cj_rem measures remaining resources
		double done;
		// Determine if episode is over (are there remaining resources?)
		if (remaining_resources < 1) {
			done = 1; //1=done
			System.out.println("No remaining resources. Episoide is finished.");
		}
		else {
			done = 0;  //0=not done
		}
		
		System.out.println("done ([0=not done, 1=done]) : ");
		System.out.println("done : " + done);
		
		return done;
	}
	
	
	public void store_qlearning_values() {
		// "Store transition (s_h, a_h, R_h, s_h+1, P_h) in experience replay buffer, where P_h = P"
		
		// Slice off last instances of state_instances for saving
		Instance lastrow = state_instances.instance(num_of_rows-1);
		
		//should only be 10 long 
		// make a vector index from 0 to 9 no matter what the cnt
		int short_cnt = VehicularEdgeOrchestrator.cnt % 10;
		
		s_t[short_cnt][0] = state_instances;
		a_t[short_cnt] = action_greedy;
		R_t[short_cnt] = R;
		y_t[short_cnt] = lastrow.value(9);  // y_actual_binary
		d_t[short_cnt] = done;
	}
	
	
	public void agent_learn(double done) {
		
		//Calculate rewards for the PREVIOUS target_q_network every update_interval (ie: every 10 counts)
		double [] Q_target = new double[update_interval];
		double [] y_target = new double[update_interval];
		double [] Q_values = new double[update_interval];
		
		if (done == 1) {
			//  No resources, stop episode
			Q_target = R_t;
			y_target = R_t;
		}
		else {
			// -----------------------------------
			// Periodically update weights
			if (VehicularEdgeOrchestrator.cnt % update_interval == 0) {
				//what is the goal?  to change the weights of the NN so that for the same type 
				// of input/state values,  it outputs the most optimal actions - sort of like GANS
				
				//Compute max Q^(s,a)
				// Then for each Instance in state0, need to evaluate then calculate 
				double max_qsa = 0;
				for(int i=0; i<update_interval; i++) {
					try {
						//Q_target[i] = target_q_network.classifyInstance(s_t[i][0].instance(i)); // Output results of model fit [0="EDGE", 1="NOEDGE"]
						//OU
						Q_target[i] = predict(s_t[i][0].instance(i), target_q_network);
						
					} catch (Exception e) {
						e.printStackTrace();
					}
					 
					max_qsa = Math.max(Q_target[i], max_qsa);  // evaluate the max at each row of the playback data
				}
				
				for(int i=0; i<update_interval; i++) {
					y_target[i] =  R_t[i] + GAMMA*max_qsa;
				}
				
				// Calculate rewards for the CURRENT Q-network
				double loss = 0;
				double sum_w_t = 0;
				double [] P_t = new double[update_interval];
				for(int i=0; i<update_interval; i++) {
					try { 
						//Q_values[i] = q_network.classifyInstance(state_instances.instance(i)); // Output results of model fit [0="EDGE", 1="NOEDGE"]
						//OU
						Q_values[i] = predict(state_instances.instance(i), q_network);
					} catch (Exception e) {
						e.printStackTrace();
					}
					// Compute the loss
					// "Calculate the absolute error w_t= abs(Q_target^u (s_t,a_t,theta') - Q^u (s_t,a_t,theta))"
					double w_t = Math.abs(y_target[i] - Q_values[i]);
					
					// "Calculate the loss function"
					loss = Double.sum(loss, Math.pow(w_t, 2));

					// "Calculate the priority of this round P_t = (np.abs(w_t)+epsilon)^nu/np.sum(np.abs(w)+epsilon)^nu"
					sum_w_t = Double.sum(sum_w_t, Math.abs(w_t)+EPSILON);
					P_t[i] = Math.pow(Math.abs(w_t)+EPSILON, NU);
					
				}
				// "Calculate the loss function" : scalar
				loss = loss/update_interval;
				
				System.out.println("loss : " + loss);
				
				// "Calculate the priority : vector
				for(int i=0; i<update_interval; i++) {
					P_t[i] = P_t[i]/Math.pow(sum_w_t, NU);
				}
				
				System.out.println("Priority vector P_t calculated");

				// "Update model parameter theta = theta - alpha*gradient_theta L(theta)"
				// Get the gradients of the loss with respect to the weights.
			
				// Outputs setting for the model: ie: learning rate, layers
				//String out [] = q_network.getOptions();
				//for (int i=0; i<out.length;i++) {
				//	System.out.println("q_network info: " + out[i]);
				//}
				
				
				try {
					q_network.next();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				// Update the weights of the q_network
				// --------------------
				// In tensorflow
				// --------------------
			    // "Update model parameter theta = theta - alpha*gradient_theta L(theta)"
			    // Get the gradients of the loss with respect to the weights.
			    // gradients = tape.gradient(loss, q_network.trainable_variables)

			    // Update the weights of the q_network.
			    // optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
			    // --------------------
				
				System.out.println("Update weight on gradients for backpropagation *** TO DO ");
				// Searched, but can not find the functionality in weka
				
			} // End of if (VehicularEdgeOrchestrator.cnt == update_interval)
		}
	}
	// ----------------------------------------------
	
	
	
	
	// ----------------------------------------------
	// Utility functions
	// ----------------------------------------------
	public Instances data_input_logger(int num_of_rows, String name_of_dfInstances, String[] attributes) {
		
		// Get number of columns (num_of_cols)
		int num_of_cols = attributes.length;
		
		// Goal: Need to put all the data (across column) in a long array, row by row
		// You make an array for the data values called instanceValue1 and an array for the column_names called attributes
		
		// ----------------------------------------------
		// Define Attributes : You only need to define the column names (it must be in an arraylist)
		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		//atts.add(new Attribute("Tall"));
		//atts.add(new Attribute("Cj_rem"));
		//atts.add(new Attribute("processing_time"));
		//atts.add(new Attribute("CPUcost"));
		//atts.add(new Attribute("BWcost"));
		//atts.add(new Attribute("packet_loss_rate"));
		//atts.add(new Attribute("getNetworkDelay"));
		//atts.add(new Attribute("result_of_task"));
		//atts.add(new Attribute("QOE"));
		//atts.add(new Attribute("y_actual_binary"));
		// OU
		// Using an ArrayList
		for(int q=0; q<num_of_cols; q++) {
			atts.add(new Attribute(attributes[q]));
		}
		// ----------------------------------------------
		
		// ----------------------------------------------
		// Define Instances :
		// Instances is just an array of values (Instances_array) and column_names (Attribute) : it is the same as a DataFrame
		// relation is just a string that gives the name of the DataFrame/Instances
		Instances state_train = new Instances(name_of_dfInstances, atts, 0); 
		// ----------------------------------------------
		
		// ----------------------------------------------
		// Read data
		String method_type = "simlogger"; //"file"
		
		if (method_type == "simlogger") {
			for(int i=0; i<num_of_rows; i++) {
				// Get sample from logger
				double[] state = {SimLogger.Tall, SimLogger.Cj_rem, SimLogger.processing_time, SimLogger.CPUcost, SimLogger.BWcost, SimLogger.packet_loss_rate, SimLogger.getNetworkDelay,SimLogger.result_of_task, SimLogger.QOE, SimLogger.y_actual_binary};
				
				// KEY: Add a sample ROW at a time to Instances!!!
				state_train.add(new DenseInstance(1.0, state));  // the one is a weight, this would only be useful for scaling or class balancing
				
				// print some streamed data
				if (i == 0) {
					System.out.println("Print a row of streamed state data:\n");
					for (int x=0;x<state.length;x++) {
						System.out.println(state[x]+ ",");
					}
					System.out.println("\n");
				}
				
			} // End of for(int i=0; i<num_of_rows; i++)
		}
		else {
			// Read from File:
			try {
				CSVLoader loader = new CSVLoader();
				String[] options = new String[5];
				options[0] = "-H";
				options[1] = "-F";
				options[2] = ",";
				options[3] = "-R";
				options[4] = "1,2,3,4,5,6,7,8,9,10-last";
				loader.setOptions(options);
				loader.setSource(new File(SimLogger.train_data_filename));
				loader.setNoHeaderRowPresent(true);
				
				state_train = loader.getDataSet();
				
			} catch (Exception e) {
				//e.printStackTrace();
			}
		}
		// ----------------------------------------------
		
		// ----------------------------------------------
		// Specify which column is y_actual/class
		// What everyone uses
		state_train.setClassIndex(state_train.numAttributes()-1);
		// int val = state_train.numAttributes()-1;
		// System.out.println("state_train.numAttributes()-1 : " + val);
		// OU
		// Specify that the last column (column 9, in a vector from 0 to 9) is y_actual_binary
		// state_train.setClassIndex(9);
		// ----------------------------------------------
		
		// ----------------------------------------------
		// Remove all instances with a missing class value
		state_train.deleteWithMissingClass();
		// ----------------------------------------------
		
		// ----------------------------------------------
		// Scale features
		//Normalize norm = new Normalize();  // normalize data to [0,1] interval
		// OU
		Standardize norm = new Standardize(); // scale data
		Instances scaled_state_train = new Instances(name_of_dfInstances, atts, 0); ;
		try {
			norm.setInputFormat(state_train);
			scaled_state_train = Filter.useFilter(state_train, norm);
		} catch (Exception e) {
			e.printStackTrace();
		}
		// ----------------------------------------------
		
		return scaled_state_train;
	}
	
	private double predict(Instance Xinstance, MultilayerPerceptron model) throws Exception {
	    double[] prediction_prob = model.distributionForInstance(Xinstance);
	    
	    // ------------ confirming the result is correct ------------
	    //System.out.println("prediction_prob.length = " + prediction_prob.length);
	    //System.out.println("prediction_prob = ");
	    for (int i=0;i<prediction_prob.length;i++){
	        System.out.println(prediction_prob[i] + " ");
	    }
	    // ------------
		double pred_class = argmax(prediction_prob);  // Output results of model fit [0="EDGE", 1="NOEDGE"]
		
		//System.out.println("pred_class : " + pred_class);
		
		return pred_class;
	}
	
	
	
	private static int argmax(double[] array) {
        double max = array[0];
        int re = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                re = i;
            }
        }
        return re;
    }
	
	// ----------------------------------------------
	
}
