`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/07/2021 03:46:51 PM
// Design Name: 
// Module Name: top_level
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

`default_nettype none
module top_level(
	input wire clk_in, rst_in,
    );
    
	// TODO: set correct clock
	logic clk_in;
	
	// bit sizes
	parameter ADDR_BITS = 32;
	parameter SIZE_BITS = 10;
	parameter LANE_BITS = 3
	parameter DATA_BITS = 8;
	
	// Layer types
	parameter DENSE = 0;
	parameter RELU = 2;
	parameter MOVE = 5;
	parameter OUTPUT = 6;
	
	// MAC steps
	parameter INIT_BIAS = 0;
	parameter MAC_OUTPUT = 1;
	
	parameter MAC_LANES = 1;
	
    // Overall layer state variables
    logic [7:0] layer_num;
    logic [2:0] layer_type;
    
    // addr bases
    logic [ADDR_BITS-1:0] input_base_addr;
    logic [ADDR_BITS-1:0] weight_base_addr;
    logic [ADDR_BITS-1:0] bias_base_addr;
    logic [ADDR_BITS-1:0] output_base_addr;

    // sizes
    logic [SIZE_BITS-1:0] m_size;
    logic [SIZE_BITS-1:0] chw_size;
    logic [SIZE_BITS-1:0] n_size;

    // linear layer overall state variables
    logic linear_layer_step;

	//AAAAAA -- some of these should be initialised to 0 to start make sure that is done 
    // linear init loop module signals
    logic [ADDR_BITS-1:0] linear_init_loop_bias_addr;
    logic [ADDR_BITS-1:0] linear_init_loop_output_addr;
	logic [ADDR_BITS-1:0] linear_init_loop_output_prev_addr;
	logic linear_init_loop_started;
	logic linear_init_loop_ready_out;
	logic linear_init_loop_done_out;
	logic linear_init_loop_start_ready_in;
	logic linear_init_loop_next_ready_in;
	logic linear_init_loop_first_val_read;
	logic [SIZE_BITS-1:0] linear_init_loop_num_writes;

    // linear mac loop module signals
	logic [ADDR_BITS-1:0] linear_mac_loop_input_addr;
	logic [ADDR_BITS-1:0] linear_mac_loop_weight_addr;
	logic [ADDR_BITS-1:0] linear_mac_loop_output_addr;
	logic linear_mac_loop_started;
	logic linear_mac_loop_ready_out;
	logic linear_mac_loop_done_out;
    logic linear_mac_loop_start_ready_in;
    logic linear_mac_loop_next_ready_in;
	logic [1:0] linear_mac_loop_read_step;
	logic [LANE_BITS-1:0] linear_mac_loop_read_lane_index;
	logic linear_mac_loop_write_step;
	logic [LANE_BITS-1:0] linear_mac_loop_write_lane_index;
	logic [ADDR_BITS-1:0] linear_mac_loop_output_addrs [MAC_LANES-1:0];  //thing to recheck that the 2D array is done correctly 

    // activation loop module signals
    logic [ADDR_BITS-1:0] activation_loop_output_addr;
	logic [ADDR_BITS-1:0] activation_loop_output_prev_addr;
	logic activation_loop_started;
	logic activation_loop_ready_out;
	logic activation_loop_done_out;
	logic activation_loop_start_ready_in;
	logic activation_loop_next_ready_in;
	logic [SIZE_BITS-1:0] activation_loop_num_reads;
	logic [SIZE_BITS-1:0] activation_loop_num_writes;
	logic activation_loop_relu_started;

	// move loop module signals
	// logic [ADDR_BITS-1:0] move_loop_bias_addr;//not needed anymore?
    logic [ADDR_BITS-1:0] move_loop_output_addr;
    logic [ADDR_BITS-1:0] move_loop_output_prev_addr;
    logic move_loop_started;
	logic move_loop_ready_out;
    logic move_loop_done_out;
	logic move_loop_start_ready_in;
	logic move_loop_next_ready_in;
	logic move_loop_first_val_read;
    logic [SIZE_BITS-1:0] move_loop_num_writes;

    // layer signals
	logic next_layer;
	
	// loop modules
	// TODO: call linear_init_loop_inst
	// TODO: call linear_mac_loop_inst
	// TODO: call relu_inst 
	// move_loop_inst = None
	// mac_inst = None

	//modules: initialised as None in python what here?
	logic linear_init_loop_inst;
	logic linear_mac_loop_inst;
    logic relu_inst;
    logic move_loop_inst;
    logic mac_inst;

	// mac module signals
	logic mac_ready_in;
	logic mac_done_out;
	logic signed [DATA_BITS-1:0] weights_mac [MAC_LANES-1:0]; //recheck initialisation of all these 
	logic signed [DATA_BITS-1:0] inputs_mac [MAC_LANES-1:0];  
	logic signed [DATA_BITS-1:0] biases_mac [MAC_LANES-1:0];  
	logic signed [DATA_BITS-1:0] outputs_mac [MAC_LANES-1:0];
	
	// relu signal
	logic relu_ready_in;
	logic relu_done_out;
	logic signed [DATA_BITS-1:0] input_relu;
	logic signed [DATA_BITS-1:0] output_relu;


	//make sure we are not confused with the bram naming especiallly as the camera ones are named similarly 
	// bram scratchpad
	logic [ADDR_BITS-1:0] bram0_read_addr;
	logic bram0_read_enable;
	logic signed [DATA_BITS-1:0] bram0_read_out;

	// bram scratchpad
	logic bram1_read_enable; //this is not needed in verilog
	logic [ADDR_BITS-1:0] bram1_read_addr;
	logic signed [DATA_BITS-1:0] bram1_read_out;
	logic bram1_write_enable;
	logic [ADDR_BITS-1:0] bram1_write_addr;
	logic bram1_write_val;

	always_ff @(posedge clk_in) begin
		if (rst_in) begin
			// TODO: initialize all variables
			relu_ready_in <= 0;
			relu_done_out <= 0; 
			input_relu <= 0;
			output_relu <= 0;
			bram0_read_addr <= 0;
			bram0_write_enable <= 0;
			bram1_read_addr <= 0;
			bram1_write_addr <= 0;
			bram1_write_enable <= 0;
		end
		else begin
			if (next_layer) begin
			
			end
			else if (layer_type == DENSE) begin
				if (linear_layer_step == INIT_BIAS) begin
					if (linear_init_loop_done_out && (linear_init_loop_num_writes == m_size)) begin
						linear_layer_step <= MAC_OUTPUT;
                        linear_init_loop_bias_addr <= 0;
                        linear_init_loop_output_addr <= 0;
                        linear_init_loop_output_prev_addr <= 0;
                        linear_init_loop_started <= 0;
                        linear_init_loop_ready_out <= 0;
                        linear_init_loop_done_out <= 0;
                        linear_init_loop_start_ready_in <= 0;
                        linear_init_loop_next_ready_in <= 0;
                        linear_init_loop_first_val_read <= 0;
                        linear_init_loop_num_writes <= 0;
					end
					
					if (linear_init_loop_next_ready_in) begin
						linear_init_loop_next_ready_in <= 0;
					end
					
					if ((linear_init_loop_num_writes < m_size) && linear_init_loop_first_val_read) begin
                        bram1_write_addr <= linear_init_loop_output_prev_addr;
                        bram1_write_enable <= 1;
                        bram1_write_val <= bram0_read_out;
                        linear_init_loop_num_writes <= linear_init_loop_num_writes + 1;
					end
                    
                    // read bias
                    if (linear_init_loop_ready_out) begin
                        bram0_read_addr <= linear_init_loop_bias_addr;
                        bram0_read_enable <= 1;
                        linear_init_loop_next_ready_in <= 1;
                        linear_init_loop_first_val_read <= 1;
                        linear_init_loop_output_prev_addr <= linear_init_loop_output_addr;
                    end
					
                    // entering this state for the first time
                    if (~linear_init_loop_started) begin
                        linear_init_loop_start_ready_in <= 1;
                        linear_init_loop_next_ready_in <= 1;
                        linear_init_loop_started <= 1;
					end
				end
				else if (linear_layer_step == MAC_OUTPUT) begin
					if (linear_mac_loop_start_ready_in) begin
						linear_mac_loop_start_ready_in <= 0;
					end
					// write output from MAC
					if (linear_mac_loop_write_step == 1) begin
						// not done writing output from MAC and MAC has finished computation
						if ((linear_mac_loop_write_lane_index < MAC_LANES) && mac_done_out) begin
							bram1_write_addr <= linear_mac_loop_output_addrs[linear_mac_loop_write_lane_index];
							bram1_write_val <= outputs_mac[linear_mac_loop_write_lane_index];
							bram1_write_enable <= 1;
							linear_mac_loop_write_lane_index <= linear_mac_loop_write_lane_index + 1;
						end
						// done writing output for MAC
						else if (linear_mac_loop_write_lane_index == MAC_LANES) begin
							linear_mac_loop_write_step <= 0; // reset to not writing state

							// done writing output and the loop is complete
							if (linear_mac_loop_done_out) begin
								next_layer <= 1; // go to next layer
							end
						end
					end
					// not finished reading all of the data for each of the MAC lates
					if (linear_mac_loop_read_lane_index < MAC_LANES) begin
						// reading weight and input from BRAM
						if ((linear_mac_loop_read_step == 0) && linear_mac_loop_ready_out) begin
							bram0_read_addr <= linear_mac_loop_weight_addr;
							bram0_read_enable <= 1;
							bram1_read_addr <= linear_mac_loop_input_addr;
							bram1_read_enable <= 1;
							linear_mac_loop_read_step <= 1;
						end
						
						// reading output from BRAM
						else if (linear_mac_loop_read_step == 1) begin
							bram0_read_enable <= 0;
							bram1_read_addr <= linear_mac_loop_output_addr; // the bias is the output in this case
							bram1_read_enable <= 1;
							weights_mac[linear_mac_loop_read_lane_index] <= bram0_read_out;
							inputs_mac[linear_mac_loop_read_lane_index] <= bram1_read_out;
							linear_mac_loop_read_step <= 2;
						end
						
						// starting MAC, and initializing next read
						else if (linear_mac_loop_read_step == 2) begin
							// if we are writing, we move on to reading the next lane only if reading is behind writing
							// otherwise if writing is complete (or not started for the first time), then we can read next lane no problems
							// this is so that we do not overwrite linear_mac_loop_output_addrs while it is being used for writing
							if ((linear_mac_loop_write_step == 1) && (linear_mac_loop_read_lane_index < linear_mac_loop_write_lane_index) || (linear_mac_loop_write_step == 0)) begin
								bram0_read_enable <= 0;
								bram1_read_enable <= 0;
								biases_mac[linear_mac_loop_read_lane_index] <= bram1_read_out;
								linear_mac_loop_output_addrs[linear_mac_loop_read_lane_index] <= linear_mac_loop_output_addr;
								// increment to next lane, and generate new addrs from loop
								linear_mac_loop_read_lane_index <= linear_mac_loop_read_lane_index + 1;
								linear_mac_loop_read_step <= 0;
								linear_mac_loop_next_ready_in <= 1;
							end
						end	
					end
					// done reading for all of the lanes
					else if (linear_mac_loop_read_lane_index == MAC_LANES) begin
						mac_ready_in <= 1; // execute mac
						linear_mac_loop_write_step <= 1; // start writing
						// restart reading from first lane
						linear_mac_loop_read_step <= 0;
						linear_mac_loop_read_lane_index <= 0; 
					end
					
					// entering this state for the first time
					if (~linear_mac_loop_started) begin
						linear_mac_loop_start_ready_in <= 1;
						linear_mac_loop_next_ready_in <= 1;
						linear_mac_loop_started <= 1;
					end
				end
			end
			else if (layer_type == RELU) begin
				
			end
			else if (layer_type == MOVE) begin
			
			end
			else if (layer_type == OUTPUT) begin
			
			end
			
		end
		
	
	end
		
		
		
	
    
endmodule

`default_nettype wire
