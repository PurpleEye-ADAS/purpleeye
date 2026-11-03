`timescale 1ns / 1ps
module traffic_pcie_control #(
    parameter integer CLK_HZ = 125000000,
    parameter integer DEBOUNCE_TICKS = CLK_HZ / 50,
    parameter integer LONG_TICKS = CLK_HZ,
    parameter integer AUTO_TICKS = CLK_HZ * 3,
    parameter integer WATCHDOG_TICKS = CLK_HZ / 2
)(
    input wire clk,
    input wire rst_n,
    input wire key_n,
    input wire command_valid,
    input wire [31:0] command,
    input wire [31:0] expected_sequence,
    output wire [127:0] status,
    output wire [1:0] led,
    output reg source_sel
);
reg key_meta, key_sync, stable_n;
reg [31:0] debounce_count, held_count, watchdog_count;
reg [1:0] navigation;
reg manual_enabled, manual_move, moving_led, alarm_led;
reg [31:0] event_sequence, command_echo;
wire [31:0] control = {28'd0, manual_enabled, manual_move, navigation};
wire key_released = !stable_n && key_sync && debounce_count == DEBOUNCE_TICKS - 1;
wire sequence_matches = expected_sequence == event_sequence;
wire accepted = command_valid && command[31:16] == 16'ha55a && !key_released &&
                (!command[6] || (sequence_matches && command[1:0] != 2'd3 &&
                 (command[7] || command[3:0] == control[3:0])));
assign status = {command_echo, event_sequence, control, 32'h54524632};
assign led = watchdog_count != 0 ? {alarm_led, moving_led} : 2'b00;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        key_meta <= 1'b1; key_sync <= 1'b1; stable_n <= 1'b1;
        debounce_count <= 0; held_count <= 0; watchdog_count <= 0;
        navigation <= 2'd1;
        manual_enabled <= 0; manual_move <= 0;
        moving_led <= 0; alarm_led <= 0;
        event_sequence <= 0; command_echo <= 0; source_sel <= 0;
    end else begin
        key_meta <= key_n;
        key_sync <= key_meta;
        if (watchdog_count != 0)
            watchdog_count <= watchdog_count - 1'b1;
        if (!stable_n && held_count < AUTO_TICKS)
            held_count <= held_count + 1'b1;
        if (key_sync == stable_n)
            debounce_count <= 0;
        else if (debounce_count == DEBOUNCE_TICKS - 1) begin
            debounce_count <= 0;
            stable_n <= key_sync;
            if (!key_sync)
                held_count <= 0;
            else begin
                event_sequence <= event_sequence + 1'b1;
                // A new vehicle state must not inherit the previous lane's permission.
                moving_led <= 0;
                alarm_led <= 0;
                watchdog_count <= 0;
                if (held_count >= AUTO_TICKS) begin
                    manual_enabled <= 0;
                    manual_move <= 0;
                end else if (held_count >= LONG_TICKS) begin
                    manual_enabled <= 1;
                    manual_move <= !manual_move;
                end else begin
                    navigation <= navigation == 2'd2 ? 2'd0 : navigation + 1'b1;
                end
                held_count <= 0;
            end
        end else
            debounce_count <= debounce_count + 1'b1;
        if (accepted) begin
            command_echo <= command;
            watchdog_count <= WATCHDOG_TICKS;
            moving_led <= command[6] && command[4];
            alarm_led <= command[6] && command[5];
            if (sequence_matches)
                source_sel <= command[8];
            if (sequence_matches && command[7] && command[1:0] != 2'd3) begin
                navigation <= command[1:0];
                manual_move <= command[2];
                manual_enabled <= command[3];
                event_sequence <= event_sequence + 1'b1;
            end
        end
    end
end
endmodule
