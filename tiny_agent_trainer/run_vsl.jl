# Julia VSL Runtime
# Executes VSL (Visual Signal Language) code safely
# Usage: julia run_vsl.jl <vsl_file>

using Printf
using LinearAlgebra

# VSL Symbol definitions
const VSL_SYMBOLS = Dict(
    "▲" => :amplifier,
    "∫" => :integrator,
    "∂" => :differentiator,
    "~" => :oscillator,
    "□" => :delay,
    "○" => :buffer,
    "△" => :filter,
    "▽" => :inverter,
    "→" => :connection,
    "←" => :feedback,
    "↑" => :upsample,
    "↓" => :downsample,
    "+" => :adder,
    "-" => :subtractor,
    "*" => :multiplier,
    "/" => :divider,
    "=" => :assignment
)

# Signal storage
signals = Dict{String, Vector{Float64}}()

# Initialize default signals
function init_signals()
    global signals
    # Create some default test signals
    t = 0:0.01:1  # Time vector
    
    signals["A1"] = sin.(2π * 1.0 * t)      # 1 Hz sine wave
    signals["A2"] = sin.(2π * 2.0 * t)      # 2 Hz sine wave
    signals["B1"] = cos.(2π * 1.0 * t)      # 1 Hz cosine wave
    signals["B2"] = ones(length(t))         # DC signal
    signals["C1"] = randn(length(t)) * 0.1  # Noise
end

# VSL Operation implementations
function vsl_amplify(input_signal::Vector{Float64}, gain::Float64 = 2.0)
    return input_signal * gain
end

function vsl_integrate(input_signal::Vector{Float64}, dt::Float64 = 0.01)
    integrated = zeros(length(input_signal))
    for i in 2:length(input_signal)
        integrated[i] = integrated[i-1] + input_signal[i] * dt
    end
    return integrated
end

function vsl_differentiate(input_signal::Vector{Float64}, dt::Float64 = 0.01)
    diff_signal = zeros(length(input_signal))
    for i in 2:length(input_signal)
        diff_signal[i] = (input_signal[i] - input_signal[i-1]) / dt
    end
    return diff_signal
end

function vsl_oscillator(frequency::Float64 = 1.0, amplitude::Float64 = 1.0, phase::Float64 = 0.0)
    t = 0:0.01:1
    return amplitude * sin.(2π * frequency * t .+ phase)
end

function vsl_delay(input_signal::Vector{Float64}, delay_samples::Int = 10)
    delayed = zeros(length(input_signal))
    for i in (delay_samples+1):length(input_signal)
        delayed[i] = input_signal[i - delay_samples]
    end
    return delayed
end

function vsl_filter_lowpass(input_signal::Vector{Float64}, cutoff::Float64 = 0.1)
    # Simple low-pass filter
    output = zeros(length(input_signal))
    alpha = cutoff
    output[1] = input_signal[1]
    
    for i in 2:length(input_signal)
        output[i] = alpha * input_signal[i] + (1 - alpha) * output[i-1]
    end
    return output
end

# Parse and execute VSL expressions
function parse_vsl_variable(var_str::String)
    # Extract variable name (e.g., "A1", "B2")
    match_result = match(r"([A-Z])(\d+)", var_str)
    if match_result !== nothing
        return var_str
    end
    return nothing
end

function parse_vsl_number(num_str::String)
    try
        return parse(Float64, num_str)
    catch
        return nothing
    end
end

function execute_vsl_operation(operation::String, args...)
    global signals
    
    if operation == "▲"  # Amplifier
        if length(args) >= 1
            var_name = string(args[1])
            gain = length(args) >= 2 ? parse(Float64, string(args[2])) : 2.0
            
            if haskey(signals, var_name)
                return vsl_amplify(signals[var_name], gain)
            else
                println("Warning: Variable $var_name not found")
                return zeros(100)
            end
        end
        
    elseif operation == "∫"  # Integrator
        if length(args) >= 1
            var_name = string(args[1])
            if haskey(signals, var_name)
                return vsl_integrate(signals[var_name])
            else
                println("Warning: Variable $var_name not found")
                return zeros(100)
            end
        end
        
    elseif operation == "∂"  # Differentiator
        if length(args) >= 1
            var_name = string(args[1])
            if haskey(signals, var_name)
                return vsl_differentiate(signals[var_name])
            else
                println("Warning: Variable $var_name not found")
                return zeros(100)
            end
        end
        
    elseif operation == "~"  # Oscillator
        frequency = length(args) >= 1 ? parse(Float64, string(args[1])) : 1.0
        amplitude = length(args) >= 2 ? parse(Float64, string(args[2])) : 1.0
        return vsl_oscillator(frequency, amplitude)
        
    elseif operation == "□"  # Delay
        if length(args) >= 1
            var_name = string(args[1])
            delay = length(args) >= 2 ? Int(parse(Float64, string(args[2]))) : 10
            
            if haskey(signals, var_name)
                return vsl_delay(signals[var_name], delay)
            else
                println("Warning: Variable $var_name not found")
                return zeros(100)
            end
        end
        
    elseif operation == "△"  # Filter
        if length(args) >= 1
            var_name = string(args[1])
            cutoff = length(args) >= 2 ? parse(Float64, string(args[2])) : 0.1
            
            if haskey(signals, var_name)
                return vsl_filter_lowpass(signals[var_name], cutoff)
            else
                println("Warning: Variable $var_name not found")
                return zeros(100)
            end
        end
        
    elseif operation == "+"  # Addition
        if length(args) >= 2
            var1 = string(args[1])
            var2 = string(args[2])
            
            if haskey(signals, var1) && haskey(signals, var2)
                return signals[var1] + signals[var2]
            else
                println("Warning: Variables $var1 or $var2 not found")
                return zeros(100)
            end
        end
        
    elseif operation == "*"  # Multiplication
        if length(args) >= 2
            var1 = string(args[1])
            var2 = string(args[2])
            
            if haskey(signals, var1) && haskey(signals, var2)
                return signals[var1] .* signals[var2]
            else
                println("Warning: Variables $var1 or $var2 not found")
                return zeros(100)
            end
        end
    end
    
    println("Warning: Unknown operation or invalid arguments: $operation")
    return zeros(100)
end

function tokenize_vsl(vsl_code::String)
    # Simple tokenizer for VSL code
    tokens = String[]
    i = 1
    
    while i <= length(vsl_code)
        char = vsl_code[i]
        
        # Skip whitespace
        if isspace(char)
            i += 1
            continue
        end
        
        # Multi-character variables (A1, B2, etc.)
        if isuppercase(char) && i < length(vsl_code) && isdigit(vsl_code[i+1])
            var_match = match(r"[A-Z]\d+", vsl_code[i:end])
            if var_match !== nothing
                push!(tokens, var_match.match)
                i += length(var_match.match)
                continue
            end
        end
        
        # Numbers
        if isdigit(char) || char == '.'
            num_match = match(r"\d+\.?\d*", vsl_code[i:end])
            if num_match !== nothing
                push!(tokens, num_match.match)
                i += length(num_match.match)
                continue
            end
        end
        
        # Single character symbols
        push!(tokens, string(char))
        i += 1
    end
    
    return tokens
end

function parse_and_execute_vsl(vsl_code::String)
    global signals
    
    println("Executing VSL code: $vsl_code")
    println("=" ^ 40)
    
    # Initialize signals if empty
    if isempty(signals)
        init_signals()
        println("Initialized default signals:")
        for (name, signal) in signals
            println("  $name: $(length(signal)) samples, range [$(minimum(signal):.3f), $(maximum(signal):.3f)]")
        end
        println()
    end
    
    tokens = tokenize_vsl(vsl_code)
    
    if isempty(tokens)
        println("Error: No tokens found in VSL code")
        return
    end
    
    println("Tokens: $tokens")
    
    # Simple parsing - look for operation patterns
    if length(tokens) >= 2
        operation = tokens[1]
        
        if operation in ["▲", "∫", "∂", "~", "□", "○", "△", "▽"]
            # Unary operations
            if length(tokens) >= 2
                if operation == "▲" && occursin("(", join(tokens))
                    # Parse amplifier with gain: ▲(A1, 2.0)
                    args_match = match(r"\(([^)]+)\)", join(tokens, ""))
                    if args_match !== nothing
                        args_str = args_match.captures[1]
                        args = [strip(arg) for arg in split(args_str, ",")]
                        result = execute_vsl_operation(operation, args...)
                        println("Result: $(length(result)) samples")
                        println("Output range: [$(minimum(result):.3f), $(maximum(result):.3f)]")
                        
                        # Store result in a new variable
                        result_var = "OUT1"
                        signals[result_var] = result
                        println("Stored result in variable: $result_var")
                        return
                    end
                end
                
                # Simple operation: ▲A1 or ∫(A1)
                arg = replace(replace(tokens[2], "(" => ""), ")" => "")
                result = execute_vsl_operation(operation, arg)
                
                if !isnothing(result)
                    println("Result: $(length(result)) samples")
                    println("Output range: [$(minimum(result):.3f), $(maximum(result):.3f)]")
                    
                    # Store result in a new variable
                    result_var = "OUT1"
                    signals[result_var] = result
                    println("Stored result in variable: $result_var")
                end
            end
            
        elseif operation in ["+", "-", "*", "/"]
            # Binary operations: A1 + B1
            if length(tokens) >= 3
                left_operand = tokens[1]
                right_operand = tokens[3]
                
                result = execute_vsl_operation(tokens[2], left_operand, right_operand)
                
                if !isnothing(result)
                    println("Result: $(length(result)) samples")
                    println("Output range: [$(minimum(result):.3f), $(maximum(result):.3f)]")
                    
                    # Store result in a new variable
                    result_var = "OUT1"
                    signals[result_var] = result
                    println("Stored result in variable: $result_var")
                end
            end
            
        elseif parse_vsl_variable(operation) !== nothing
            # Just displaying a variable
            var_name = operation
            if haskey(signals, var_name)
                signal = signals[var_name]
                println("Variable $var_name:")
                println("  Length: $(length(signal)) samples")
                println("  Range: [$(minimum(signal):.3f), $(maximum(signal):.3f)]")
                println("  RMS: $(sqrt(mean(signal.^2)):.3f)")
            else
                println("Error: Variable $var_name not found")
            end
        end
    end
    
    println("\nAvailable signals:")
    for (name, signal) in sort(collect(signals))
        println("  $name: $(length(signal)) samples")
    end
end

function main()
    if length(ARGS) < 1
        println("Usage: julia run_vsl.jl <vsl_file>")
        println("Example: julia run_vsl.jl script.vsl")
        exit(1)
    end
    
    vsl_file = ARGS[1]
    
    if !isfile(vsl_file)
        println("Error: VSL file '$vsl_file' not found")
        exit(1)
    end
    
    try
        vsl_code = read(vsl_file, String)
        vsl_code = strip(vsl_code)
        
        if isempty(vsl_code)
            println("Error: VSL file is empty")
            exit(1)
        end
        
        println("VSL Runtime - Julia Implementation")
        println("Processing file: $vsl_file")
        println()
        
        parse_and_execute_vsl(vsl_code)
        
        println("\n✅ VSL execution completed successfully")
        
    catch e
        println("❌ Error executing VSL code:")
        println(e)
        exit(1)
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
