globalThis.Transformer = function(tIndex, L_param) {
	const dimensions = NetworkMeta.dimensions;
	const headDim = dimensions / NetworkMeta.heads;
	let L = L_param;

	this.setInputs = (inputs, L_param) => {
		L = L_param;
		this.inputs = lmNetwork.setInputs_WEBGPU(inputs, L);
	}

	this.RMSNorm = (xInputs, rmsGamma) => {
		const xInputsPostRMSGamma = lmNetwork.RMSNorm_WEBGPU(xInputs, rmsGamma, L);
		return xInputsPostRMSGamma;
	}

	this.generateVals = () => {
		this.vals = lmNetwork.matMul_dim_L_dim_dim_WEBGPU(this.valueWeights, this.inputsPostRMS, L);
		this.valsByHead = this.vals;
	}

	this.generateKeys = () => {
		this.keys = lmNetwork.matMul_dim_L_dim_dim_WEBGPU(this.keyWeights, this.inputsPostRMS, L);
		this.keysByHead = this.keys;
	}

	this.generateQueries = () => {
		this.queries = lmNetwork.matMul_dim_L_dim_dim_WEBGPU(this.queryWeights, this.inputsPostRMS, L);
		this.queriesByHead = this.queries;
	}

	this.applyRoPE = (keysOrQueriesByHead) => {
		if (!lmNetwork.precomputedTheta) {
			const flatPrecomputedTheta = new Float32Array(headDim * L);

			const headPairs = headDim / 2;
			for (let colIndex = 0; colIndex < L; colIndex++) {
				for (let pairIndex = 0; pairIndex < headPairs; pairIndex++) {
					flatPrecomputedTheta[colIndex * headDim + pairIndex * 2] = Math.cos(colIndex / Math.pow(NetworkMeta.ropeDenomBase, (2 * pairIndex / headDim)));
					flatPrecomputedTheta[colIndex * headDim + pairIndex * 2 + 1] = (Math.sin(colIndex / Math.pow(NetworkMeta.ropeDenomBase, (2 * pairIndex / headDim))));
				}
			}
			
			lmNetwork.precomputedTheta = executeLoadPrecomputedTheta(lmNetwork, flatPrecomputedTheta, headPairs * L, 2);
		}

		const keysOrQueriesByHeadPostRoPE = lmNetwork.RoPE_WEBGPU(keysOrQueriesByHead, lmNetwork.precomputedTheta);
		return keysOrQueriesByHeadPostRoPE;
	}

	this.getAttentionScoresByHead = () => {	
		const attentionByHeadPreSoftmax = lmNetwork.matMul_KtQ_WEBGPU(this.keysByHeadPostRoPE, this.queriesByHeadPostRoPE);
		const attentionByHeadPreSoftmaxScaledMasked = lmNetwork.scaleAndMaskAttention_WEBGPU(attentionByHeadPreSoftmax);
		this.attentionByHeadPostSoftmax = lmNetwork.softmaxByHead_WEBGPU(attentionByHeadPreSoftmaxScaledMasked);
		this.valueScaledAttentionByHead = lmNetwork.matMulValsAttention_WEBGPU(this.valsByHead, this.attentionByHeadPostSoftmax);
	}

	this.concatAttentionScoreHeads = () => {
		this.concatAllHeadAttentionScores = lmNetwork.concatHeads_WEBGPU(this.valueScaledAttentionByHead);
	}

	this.projectConcatAttentionToOutput = () => {
		this.outputProjectedAttention = lmNetwork.matMul_dim_L_dim_dim_WEBGPU(this.outputProjectionWeights, this.concatAllHeadAttentionScores);
	}

	this.addResidualInputsToConcatAttentionOutput = () => {
		this.outputProjectedAttentionScoresPlusResidualInputs = lmNetwork.elementWiseAdd_WEBGPU(this.outputProjectedAttention, this.inputs);
	}

	this.applyFeedforwardToAttention = () => {
		const ffnDim = dimensions * NetworkMeta.ffnDimMultiplier;

		const left1MatrixPreSilu = lmNetwork.matMul_FFN1_WEBGPU(this.feedForwardWeights1A, this.outputProjectedAttentionScoresPostRMS2);
		const left1Matrix = lmNetwork.applySilu_WEBGPU(left1MatrixPreSilu);
		const right1Matrix = lmNetwork.matMul_FFN1_WEBGPU(this.feedForwardWeights1B, this.outputProjectedAttentionScoresPostRMS2);

		const final1Matrix = lmNetwork.applyHadamard_WEBGPU(left1Matrix, right1Matrix);
		
		this.attentionPostFeedForward = lmNetwork.matMul_FFN2_WEBGPU(this.feedForwardWeights2, final1Matrix);
	}

	this.getFinalTransformerOutputByAddingResidualToFFNResult = () => {
		this.finalTransformerOutput = lmNetwork.elementWiseAdd_WEBGPU(this.attentionPostFeedForward, this.outputProjectedAttentionScoresPlusResidualInputs);
	}

	this.applyTransformChainToInputs = (downstreamInputs) => {
		this.inputs = downstreamInputs;
		this.inputsPostRMS = this.RMSNorm(this.inputs, this.rmsGamma, 'rms1');

		this.generateVals();
		this.generateKeys();
		this.generateQueries();

		this.keysByHeadPostRoPE = this.applyRoPE(this.keysByHead);
		this.queriesByHeadPostRoPE = this.applyRoPE(this.queriesByHead);

		this.getAttentionScoresByHead();
		this.concatAttentionScoreHeads();
		this.projectConcatAttentionToOutput();
		this.addResidualInputsToConcatAttentionOutput();

		this.outputProjectedAttentionScoresPostRMS2 = this.RMSNorm(this.outputProjectedAttentionScoresPlusResidualInputs, this.rmsGamma2, 'rms2');
		
		this.applyFeedforwardToAttention();
		this.getFinalTransformerOutputByAddingResidualToFFNResult();

		return this.finalTransformerOutput;
	}
}

globalThis.LmNetwork = function () {
  	this.init = async () => {
    	const adapter = await navigator.gpu.requestAdapter();
    	if (!adapter) throw new Error("WebGPU adapter not available");
    	this.webgpuDevice = await adapter.requestDevice();
  	}
	
	this.setShaders = () => {	
		if (!this.load_dim_dim_weights_WEBGPU) {
			const dimDimShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let dimensions = ${dimensions}u;
					
					if (col < dimensions && row < dimensions) {
						let index = row * dimensions + col;
						output[index] = input[index];
					}
				}
			`;

			this.load_dim_dim_weights_WEBGPU = function(weightMatrix) {			
				const flatInput = new Float32Array(dimensions * dimensions);
				for (let row = 0; row < dimensions; row++) {
					for (let col = 0; col < dimensions; col++) {
						flatInput[row * dimensions + col] = weightMatrix[row][col];
					}
				}

				return executeDimDimWeightLoading(this, flatInput, dimensions, dimDimShaderCode);
			};
		}

		if (!this.load_rms_weights_WEBGPU) {
			const rmsShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(64)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let index = global_id.x;
					let dimensions = ${dimensions}u;
					
					if (index < dimensions) {
						output[index] = input[index];
					}
				}
			`;

			this.load_rms_weights_WEBGPU = function(weightMatrix) {
				const flatInput = new Float32Array(weightMatrix);		
				return executeRmsWeightLoading(this, flatInput, dimensions, rmsShaderCode);
			};
		}
		
		if (!this.load_ffn1_weights_WEBGPU) {
			const ffn1ShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let numCols = ${dimensions}u;
					let numRows = ${dimensions * NetworkMeta.ffnDimMultiplier}u;
					
					if (col < numCols && row < numRows) {
						let index = row * numCols + col;
						output[index] = input[index];
					}
				}
			`;

			this.load_ffn1_weights_WEBGPU = function(weightMatrix) {
				const numRows = dimensions * NetworkMeta.ffnDimMultiplier;
				const numCols = dimensions;
				
				const flatInput = new Float32Array(numRows * numCols);
				for (let row = 0; row < numRows; row++) {
					for (let col = 0; col < numCols; col++) {
						flatInput[row * numCols + col] = weightMatrix[row][col];
					}
				}

				return executeFfn1WeightLoading(this, flatInput, numRows, numCols, ffn1ShaderCode);
			};
		}
		
		if (!this.load_ffn2_weights_WEBGPU) {
			const ffn2ShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let numCols = ${dimensions * NetworkMeta.ffnDimMultiplier}u;
					let numRows = ${dimensions}u;
					
					if (col < numCols && row < numRows) {
						let index = row * numCols + col;
						output[index] = input[index];
					}
				}
			`;

			this.load_ffn2_weights_WEBGPU = function(weightMatrix) {
				const numRows = dimensions;
				const numCols = dimensions * NetworkMeta.ffnDimMultiplier;
				
				const flatInput = new Float32Array(numRows * numCols);
				for (let row = 0; row < numRows; row++) {
					for (let col = 0; col < numCols; col++) {
						flatInput[row * numCols + col] = weightMatrix[row][col];
					}
				}

				return executeFfn2WeightLoading(this, flatInput, numRows, numCols, ffn2ShaderCode);
			};
		}

		if (!this.load_token_embeddings_WEBGPU) {
			const numRows = vocabSize;
			const numCols = dimensions;

			const tokenEmbeddingsShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let numCols = ${numCols}u;
					let numRows = ${numRows}u;
					
					if (col < numCols && row < numRows) {
						let inputIndex = row * numCols + col;
						let outputIndex = row * numCols + col;
						output[outputIndex] = input[inputIndex];
					}
				}
			`;

			this.load_token_embeddings_WEBGPU = function(weightMatrix) {			
				const flatInput = new Float32Array(numRows * numCols);
				for (let row = 0; row < numRows; row++) {
					for (let col = 0; col < numCols; col++) {
						flatInput[row * numCols + col] = weightMatrix[row][col];
					}
				}

				return executeTokenEmbeddingsWeightLoading(this, flatInput, numRows, numCols, tokenEmbeddingsShaderCode);
			};
		}

		if (!this.setInputTokenEmbeddings_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> indices: array<u32>;
				@group(0) @binding(1) var<storage, read> tokenEmbeddings: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // dimensions
					let dimensions = ${dimensions}u;
					let L = ${L}u;
					
					if (col < L && row < dimensions) {
						let tokenIndex = indices[col];
						let embeddingIndex = tokenIndex * dimensions + row;
						let outputIndex = row * L + col;
						output[outputIndex] = tokenEmbeddings[embeddingIndex];
					}
				}
			`;

			this.setInputTokenEmbeddings_WEBGPU = function(inputTokenEmbeddingIndices, tokenEmbeddings) {			
				const indicesArray = new Uint32Array(inputTokenEmbeddingIndices);
				return executeSetInputTokenEmbeddings(this, indicesArray, tokenEmbeddings.buffer, dimensions, L, shaderCode);
			};
		}

		if (!this.setInputs_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> inputs: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let dimensions = ${dimensions}u;
					let L = ${L}u;
					
					if (col < L && row < dimensions) {
						let index = row * L + col;
						output[index] = inputs[index];
					}
				}
			`;

			this.setInputs_WEBGPU = function(xInputsBuffer) {
				return executeSetInputs(this, xInputsBuffer.buffer, dimensions, L, shaderCode);
			};
		}

		if (!this.RMSNorm_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> xInputs: array<f32>;
				@group(0) @binding(1) var<storage, read> rmsGamma: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let dimensions = ${dimensions}u;
					let L = ${L}u;
					
					if (col < L && row < dimensions) {
						var colSquareSum: f32 = 0.0;
						for (var rowIndex: u32 = 0u; rowIndex < dimensions; rowIndex = rowIndex + 1u) {
							let index = rowIndex * L + col;
							let val = xInputs[index];
							colSquareSum = colSquareSum + (val * val);
						}
						
						let denominator = sqrt((colSquareSum / f32(dimensions)) + 1e-8);
						
						let index = row * L + col;
						let normalized = xInputs[index] / denominator;
						output[index] = normalized * rmsGamma[row];
					}
				}
			`;

			this.RMSNorm_WEBGPU = function(xInputs, rmsGamma) {
				return executeRMSNorm(this, xInputs.buffer, rmsGamma.buffer, dimensions, L, shaderCode);
			};
		}

		if (!this.matMul_dim_L_dim_dim_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> a: array<f32>;
				@group(0) @binding(1) var<storage, read> b: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // dimensions
					let dimensions = ${dimensions}u;
					let L = ${L}u;
					
					if (col < L && row < dimensions) {
						var sum: f32 = 0.0;
						for (var i: u32 = 0u; i < dimensions; i = i + 1u) {
							let aIndex = row * dimensions + i;
							let bIndex = i * L + col;
							sum = sum + (a[aIndex] * b[bIndex]);
						}
						let outputIndex = row * L + col;
						output[outputIndex] = sum;
					}
				}
			`;

			this.matMul_dim_L_dim_dim_WEBGPU = function(a, b) {
				return executeMatMul_dim_L_dim_dim(this, a.buffer, b.buffer, dimensions, L, shaderCode);
			};
		}

		if (!this.splitVKQByHead_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> fullKQV: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // headDim
					let head = global_id.z; // head index
					let headDim = ${headDim}u;
					let L = ${L}u;
					let heads = ${heads}u;
					
					if (col < L && row < headDim && head < heads) {
						let inputRow = head * headDim + row;
						let inputIndex = inputRow * L + col;
						let headOffset = head * headDim * L;
						let outputIndex = headOffset + row * L + col;
						output[outputIndex] = fullKQV[inputIndex];
					}
				}
			`;

			this.splitVKQByHead_WEBGPU = function(fullKQV) {
				return executeSplitVKQByHead(this, fullKQV.buffer, headDim, heads, L, shaderCode);
			};
		}
	 
		if (!this.RoPE_WEBGPU) {
			const headPairs = headDim / 2;

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read> theta: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // headDim
					let head = global_id.z; // head index
					
					let L = ${L}u;
					let headDim = ${headDim}u;
					let heads = ${heads}u;
					let headPairs = ${headPairs}u;
					
					if (col >= L || row >= headDim || head >= heads) {
						return;
					}
					
					let headOffset = head * headDim * L;
					let pairIndex = row / 2u;
					
					// theta lookup: (col * headPairs + pairIndex) * 2 + (0 for cos, 1 for sin)
					let thetaBase = (col * headPairs + pairIndex) * 2u;
					let cosVal = theta[thetaBase];
					let sinVal = theta[thetaBase + 1u];

					var x: f32;
					var y: f32;
					if (row % 2u == 0u) {
						// even row: x = current, y = next row
						x = input[headOffset + row * L + col];
						y = input[headOffset + (row + 1u) * L + col];
						// even: cos * x - sin * y
						output[headOffset + row * L + col] = cosVal * x - sinVal * y;
					} else {
						// odd row: x = previous row, y = current
						x = input[headOffset + (row - 1u) * L + col];
						y = input[headOffset + row * L + col];
						// odd: sin * x + cos * y
						output[headOffset + row * L + col] = sinVal * x + cosVal * y;
					}
				}
			`;

			this.RoPE_WEBGPU = function(keysOrQueriesByHead, precomputedTheta) {
				return executeRoPE(this, keysOrQueriesByHead.buffer, precomputedTheta.buffer, headDim, heads, L, shaderCode);
			};
		}

		if (!this.matMul_KtQ_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> keys: array<f32>;
				@group(0) @binding(1) var<storage, read> queries: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let outCol = global_id.x; // L
					let outRow = global_id.y; // L
					let head = global_id.z; // head index
					
					let L = ${L}u;
					let headDim = ${headDim}u;
					let heads = ${heads}u;
					let inputHeadOffset = head * headDim * L;

					if (outCol >= L || outRow >= L || head >= heads) {
						return;
					}
					
					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < headDim; i = i + 1u) {
						let kIndex = inputHeadOffset + i * L + outRow; // iterate over a column (transposed row)
						let qIndex = inputHeadOffset + i * L + outCol; // also over column on the right matrix (as always)
						sum = sum + keys[kIndex] * queries[qIndex];
					}
					
					let outputHeadOffset = head * L * L;
					let outputIndex = outputHeadOffset + outRow * L + outCol;
					output[outputIndex] = sum;
				}
			`;

			this.matMul_KtQ_WEBGPU = function(keys, queries) {
				return executeKtQ(this, keys.buffer, queries.buffer, headDim, heads, L, shaderCode);
			};
		}

		if (!this.scaleAndMaskAttention_WEBGPU) {
			const sqrtHeadDim = Math.sqrt(headDim);

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;

				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // column L
					let row = global_id.y; // row L
					let head = global_id.z; // head index
					
					let L = ${L}u;
					let heads = ${heads}u;
					let sqrtHeadDim = ${sqrtHeadDim}f;
					
					if (col >= L || row >= L || head >= heads) {
						return;
					}
					
					let headOffset = head * L * L;
					let index = headOffset + row * L + col;
					
					// causal mask: if row > col, set to -Infinity
					if (row > col) {
						// a very large negative number
						output[index] = -3.402823e+38f;
					} else {
						output[index] = input[index] / sqrtHeadDim;
					}
				}
			`;

			this.scaleAndMaskAttention_WEBGPU = function(attentionScores) {
				return executeScaleAndMask(this, attentionScores.buffer, heads, L, shaderCode);
			};
		}

		if (!this.softmaxByHead_WEBGPU) {
			const colMaxShader = `
			@group(0) @binding(0) var<storage, read> attnScores: array<f32>;
			@group(0) @binding(1) var<storage, read_write> maxByCol: array<f32>;
			@compute @workgroup_size(32)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				const L = ${L}u;
				const heads = ${heads}u;

				let globalCol = global_id.x; // L

				if (globalCol >= L * heads) {
					return;
				}

				let headIndex = globalCol / L; // u type drops decimal automatically
				let colWithinHeadIndex = globalCol - headIndex * L;

				let attnColIndex = headIndex * L * L + colWithinHeadIndex;
				var max: f32 = -3.402823e+38f;
				for (var i: u32 = 0u; i <= colWithinHeadIndex; i = i + 1u) {
					if (attnScores[attnColIndex + i * L] > max) {
						max = attnScores[attnColIndex + i * L];
					}
				}

				maxByCol[globalCol] = max;
			}
			`

			const colSumShader = `
			@group(0) @binding(0) var<storage, read> attnScores: array<f32>;
			@group(0) @binding(1) var<storage, read> maxByCol: array<f32>;
			@group(0) @binding(2) var<storage, read_write> sumByCol: array<f32>;
			@compute @workgroup_size(32)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				const L = ${L}u;
				const heads = ${heads}u;

				let globalCol = global_id.x; // L

				if (globalCol >= L * heads) {
					return;
				}

				let headIndex = globalCol / L; // u type drops decimal automatically
				let colWithinHeadIndex = globalCol - headIndex * L;

				let attnColIndex = headIndex * L * L + colWithinHeadIndex;
				let colMax = maxByCol[globalCol];
				var sum: f32 = 0f;
				for (var i: u32 = 0u; i <= colWithinHeadIndex; i = i + 1u) {
					sum += exp(attnScores[attnColIndex + i * L] - colMax);
				}

				sumByCol[globalCol] = sum;
			}
			`

			const colSoftmaxShader = `
			@group(0) @binding(0) var<storage, read> attnScores: array<f32>;
			@group(0) @binding(1) var<storage, read> maxByCol: array<f32>;
			@group(0) @binding(2) var<storage, read> sumByCol: array<f32>;
			@group(0) @binding(3) var<storage, read_write> softmaxAttnScores: array<f32>;
			@compute @workgroup_size(8, 8, 1)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				const L = ${L}u;
				const heads = ${heads}u;

				let headCol = global_id.x; // L
				let headRow = global_id.y; // L
				let headIndex = global_id.z;

				if (headCol >= L || headRow >= L || headIndex >= heads) {
					return;
				}

				let globalColIndex = L * headIndex + headCol;
				let attnScoreIndex = headIndex * L * L + headRow * L + headCol;

				// masking
				if (headRow > headCol) {
					softmaxAttnScores[attnScoreIndex] = 0.0;
					return;
				}

				let expSumOfCol = sumByCol[globalColIndex];
				softmaxAttnScores[attnScoreIndex] = exp(attnScores[attnScoreIndex] - maxByCol[globalColIndex]) / expSumOfCol;
			}			
			`

			this.softmaxByHead_WEBGPU = function(attentionScores) {
				const colMax = executeColMax(this, attentionScores.buffer, heads, L, colMaxShader);
				const colSum = executeColSum(this, attentionScores.buffer, colMax.buffer, heads, L, colSumShader);
				return executeSoftmaxByHead(this, attentionScores.buffer, colMax.buffer, colSum.buffer, heads, L, colSoftmaxShader);
			}
		}

		if (!this.matMulValsAttention_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> vals: array<f32>;
				@group(0) @binding(1) var<storage, read> attention: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let outCol = global_id.x; // L
					let outRow = global_id.y; // L
					let head = global_id.z; // head index
					
					let L = ${L}u;
					let headDim = ${headDim}u;
					let heads = ${heads}u;
					
					if (outCol >= L || outRow >= headDim || head >= heads) {
						return;
					}
					
					// head offsets
					let valsHeadOffset = head * headDim * L;
					let attHeadOffset = head * L * L;
					
					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < L; i = i + 1u) {
						let vIndex = valsHeadOffset + outRow * L + i;
						let aIndex = attHeadOffset + i * L + outCol;
						sum = sum + vals[vIndex] * attention[aIndex];
					}
					
					let outputIndex = valsHeadOffset + outRow * L + outCol;
					output[outputIndex] = sum;
				}
			`;

			this.matMulValsAttention_WEBGPU = function(vals, attention) {
				return executeMatMulValsAttention(this, vals.buffer, attention.buffer, headDim, heads, L, shaderCode);
			};
		}

		if (!this.concatHeads_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // dimensions
					
					let L = ${L}u;
					let dimensions = ${dimensions}u;
					let headDim = ${headDim}u;
					
					if (col >= L || row >= dimensions) {
						return;
					}
					
					let head = row / headDim;
					let rowWithinHead = row % headDim;
					
					let inputIndex = head * headDim * L + rowWithinHead * L + col;
					let outputIndex = row * L + col;
					output[outputIndex] = input[inputIndex];
				}
			`;

			this.concatHeads_WEBGPU = function(valueScaledAttentionByHead) {
				return executeConcatHeads(this, valueScaledAttentionByHead.buffer, dimensions, headDim, heads, L, shaderCode);
			};
		}

		if (!this.elementWiseAdd_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> a: array<f32>;
				@group(0) @binding(1) var<storage, read> b: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // dimensions
					
					let L = ${L}u;
					let dimensions = ${dimensions}u;
					
					if (col >= L || row >= dimensions) {
						return;
					}
					
					let index = row * L + col;
					output[index] = a[index] + b[index];
				}
			`;

			this.elementWiseAdd_WEBGPU = function(a, b) {
				return executeElementWiseAdd(this, a.buffer, b.buffer, dimensions, L, shaderCode);
			};
		}

		if (!this.matMul_FFN1_WEBGPU) {
			const ffnDim = dimensions * NetworkMeta.ffnDimMultiplier;

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> weights: array<f32>;
				@group(0) @binding(1) var<storage, read> input: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // ffnDim
					
					let L = ${L}u;
					let dimensions = ${dimensions}u;
					let ffnDim = ${ffnDim}u;
					
					if (col >= L || row >= ffnDim) {
						return;
					}
					
					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < dimensions; i = i + 1u) {
						let wIndex = row * dimensions + i;
						let iIndex = i * L + col;
						sum = sum + weights[wIndex] * input[iIndex];
					}
					
					let outputIndex = row * L + col;
					output[outputIndex] = sum;
				}
			`;

			this.matMul_FFN1_WEBGPU = function(weights, input) {
				return executeMatMulFFN1(this, weights.buffer, input.buffer, ffnDim, dimensions, L, shaderCode);
			};
		}
		
		if (!this.applySilu_WEBGPU) {
			const ffnDim = NetworkMeta.dimensions * NetworkMeta.ffnDimMultiplier;
			
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let L = ${L}u;
					let ffnDim = ${ffnDim}u;
					
					if (col >= L || row >= ffnDim) {
						return;
					}
					
					let idx = row * L + col;
					let x = input[idx];
					// SiLU: x / (1 + exp(-x))
					output[idx] = x / (1.0 + exp(-x));
				}
			`;

			this.applySilu_WEBGPU = function(input) {
				return executeSilu(this, input.buffer, ffnDim, L, shaderCode);
			};
		}

		if (!this.applyHadamard_WEBGPU) {
			const ffnDim = NetworkMeta.dimensions * NetworkMeta.ffnDimMultiplier;
			
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> left: array<f32>;
				@group(0) @binding(1) var<storage, read> right: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let L = ${L}u;
					let ffnDim = ${ffnDim}u;
					
					if (col >= L || row >= ffnDim) {
						return;
					}
					
					let idx = row * L + col;
					output[idx] = left[idx] * right[idx];
				}
			`;

			this.applyHadamard_WEBGPU = function(left, right) {
				return executeHadamard(this, left.buffer, right.buffer, ffnDim, L, shaderCode);
			};
		}

		if (!this.matMul_FFN2_WEBGPU) {
			const ffnDim = NetworkMeta.dimensions * NetworkMeta.ffnDimMultiplier;

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> weights: array<f32>;
				@group(0) @binding(1) var<storage, read> input: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // dimensions
					let L = ${L}u;
					let dimensions = ${dimensions}u;
					let ffnDim = ${ffnDim}u;
					
					if (col >= L || row >= dimensions) {
						return;
					}

					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < ffnDim; i = i + 1u) {
						let wIndex = row * ffnDim + i; // weights[row, i]
						let iIndex = i * L + col; // input[i, col]
						sum = sum + weights[wIndex] * input[iIndex];
					}
					
					let outputIndex = row * L + col;
					output[outputIndex] = sum;
				}
			`;

			this.matMul_FFN2_WEBGPU = function(weights, input) {
				return executeMatMulFFN2(this, weights.buffer, input.buffer, dimensions, ffnDim, L, shaderCode);
			}
		}

		if (!this.matMul_vocab_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
				@group(0) @binding(1) var<storage, read> input: array<f32>;
				@group(0) @binding(2) var<storage, read> teacherModeArr: array<u32>;
				@group(0) @binding(3) var<storage, read> rightEndIndexArr: array<u32>;
				@group(0) @binding(4) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // vocabSize
					let L = ${L}u;
					let vocabSize = ${vocabSize}u;
					let dimensions = ${dimensions}u;
					
					if (col >= L || row >= vocabSize) {
						return;
					}

					let outputIndex = row * L + col;

					if (teacherModeArr[0] == 0 && col != rightEndIndexArr[0]) {
						output[outputIndex] = 0.0;
						return;	
					}
					
					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < dimensions; i = i + 1u) {
						let eIndex = row * dimensions + i; // embeddings[row, i]
						let iIndex = i * L + col; // input[i, col]
						sum = sum + embeddings[eIndex] * input[iIndex];
					}
					
					output[outputIndex] = sum;
				}
			`;

			this.matMul_vocab_WEBGPU = function(embeddings, input) {		
				return executeMatMulVocab(this, embeddings.buffer, input.buffer, vocabSize, dimensions, L, shaderCode);
			};
		}

		if (!this.logitSoftmax_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> logits: array<f32>;
				@group(0) @binding(1) var<storage, read> teacherModeArr: array<u32>;
				@group(0) @binding(2) var<storage, read> rightEndIndexArr: array<u32>;				
				@group(0) @binding(3) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x; // L
					let row = global_id.y; // vocab index
					let L = ${L}u;
					let vocabSize = ${vocabSize}u;
					
					if (col >= L || row >= vocabSize) {
						return;
					}

					let outputIndex = row * L + col;

					if (teacherModeArr[0] == 0 && col != rightEndIndexArr[0]) {
						output[outputIndex] = 0.0;
						return;	
					}

					var maxVal: f32 = -1e38;
					for (var i: u32 = 0u; i < vocabSize; i = i + 1u) {
						let val = logits[i * L + col];
						if (val > maxVal) {
							maxVal = val;
						}
					}
					
					var expSum: f32 = 0.0;
					for (var i: u32 = 0u; i < vocabSize; i = i + 1u) {
						expSum = expSum + exp(logits[i * L + col] - maxVal);
					}					
					if (expSum < 1e-8) {
						expSum = 1e-8;
					}
					
					output[outputIndex] = exp(logits[outputIndex] - maxVal) / expSum;
				}
			`;

			this.logitSoftmax_WEBGPU = function(logits) {
				return executeLogitSoftmax(this, logits.buffer, vocabSize, L, shaderCode);
			};
		}

		if (!this.extractPredictionsAtPosition_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> softmax: array<f32>;
				@group(0) @binding(1) var<storage, read> rightEndIndexArr: array<u32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(64, 1, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let vocabIdx = global_id.x;
					let vocabSize = ${vocabSize}u;
					let L = ${L}u;
					let rightEndIndex = rightEndIndexArr[0];
					
					if (vocabIdx >= vocabSize) {
						return;
					}
					
					// [vocabSize, 2]
					// 0: vocab index, 1: probability
					let outIdx0 = vocabIdx * 2u; // index column
					let outIdx1 = vocabIdx * 2u + 1u; // probability column
					
					output[outIdx0] = f32(vocabIdx);
					output[outIdx1] = softmax[vocabIdx * L + rightEndIndex];
				}
			`;

			this.extractPredictionsAtPosition_WEBGPU = function(softmaxOutput, rightEndIndex) {
				return executeExtractPredictions(this, softmaxOutput.buffer, vocabSize, L, rightEndIndex, shaderCode);
			};
		}
	}

	this.setTransformers = () => {	
		this.transformers = [];
		for (let tIndex = 0; tIndex < NetworkMeta.numTransformers; tIndex++) {
			this.transformers.push(new Transformer(tIndex, L));
			
			this.transformers[this.transformers.length - 1].queryWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].queryWeights);
			this.transformers[this.transformers.length - 1].keyWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].keyWeights);
			this.transformers[this.transformers.length - 1].valueWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].valueWeights);
			
			this.transformers[this.transformers.length - 1].rmsGamma = this.load_rms_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].rmsGamma);
			this.transformers[this.transformers.length - 1].rmsGamma2 = this.load_rms_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].rmsGamma2);
			
			this.transformers[this.transformers.length - 1].outputProjectionWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].outputProjectionWeights);
			
			this.transformers[this.transformers.length - 1].feedForwardWeights1A = this.load_ffn1_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].feedForwardWeights1A);
			this.transformers[this.transformers.length - 1].feedForwardWeights1B = this.load_ffn1_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].feedForwardWeights1B);
			
			this.transformers[this.transformers.length - 1].feedForwardWeights2 = this.load_ffn2_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].feedForwardWeights2);
		}
	}

	const dimensions = NetworkMeta.dimensions;
	const heads = NetworkMeta.heads;
	const headDim = NetworkMeta.dimensions / NetworkMeta.heads;
	let L = -1;
	this.tokenVals = tokenVal_GLOBAL;
	this.tokenValsRecord = {};
	for (let i = 0; i < this.tokenVals.length; i++) {
		this.tokenValsRecord[this.tokenVals[i]] = i;
	}
	this.tokenEmbeddings = tokenEmbeddings_GLOBAL;
	this.vocabSize = this.tokenEmbeddings.length;
	const vocabSize = this.vocabSize;

	this.setupInference = (sequenceLength) => {
		L = sequenceLength;
		this.setShaders();
		this.setTransformers();

		this.tokenEmbeddings = this.load_token_embeddings_WEBGPU(this.tokenEmbeddings);
		this.rmsGamma3 = this.load_rms_weights_WEBGPU(rmsGamma3_GLOBAL);		
	}

	this.runQueryThroughModel = (queryArr, nextCorrectTokenIndexToPredict, teacherMode = true, rightEndIndex = -1) => {
		const startInferenceCycle = performance.now();

		globalThis.commandEncoder = this.webgpuDevice.createCommandEncoder();
		globalThis.passEncoder = commandEncoder.beginComputePass();

		globalThis.executeSetTeacherMode(this, teacherMode);
		globalThis.executeSetRightEndIndex(this, rightEndIndex);

		this.inputTokenEmbeddingIndices = [];
		this.tokenOrderInInputSequenceByEmbeddingIndex = {};
		for (let colIndex = 0; colIndex < L; colIndex++) {
			const tokenIndexInVocab = this.tokenValsRecord[queryArr[colIndex]];

			this.inputTokenEmbeddingIndices.push(tokenIndexInVocab);

			if (!this.tokenOrderInInputSequenceByEmbeddingIndex[tokenIndexInVocab]) {
				this.tokenOrderInInputSequenceByEmbeddingIndex[tokenIndexInVocab] = [colIndex];
			} else {
				this.tokenOrderInInputSequenceByEmbeddingIndex[tokenIndexInVocab].push(colIndex);
			}
		}
		this.inputTokenEmbeddings = this.setInputTokenEmbeddings_WEBGPU(this.inputTokenEmbeddingIndices, this.tokenEmbeddings);

		this.transformToGetNextToken();
	
		if (!teacherMode) {
			// Input: [vocabSize, L], Output: [vocabSize, 2] (index, probability)
			this.rightEndIndexTokenSoftmaxPredictions = this.extractPredictionsAtPosition_WEBGPU(
				this.logitScoresByInputTokenPostSoftmax, rightEndIndex
			);
		}

		globalThis.passEncoder.end();
		this.webgpuDevice.queue.submit([globalThis.commandEncoder.finish()]);

		const endInferenceCycle = performance.now();
		console.log(`GPU Inference Execution time: ${endInferenceCycle - startInferenceCycle} ms`);	
	}

	this.transformToGetNextToken = () => {
		let nextTransformerInput = this.inputTokenEmbeddings;

		for (let tIndex = NetworkMeta.numTransformers - 1; tIndex >= 0; tIndex--) {
			nextTransformerInput = this.transformers[tIndex].applyTransformChainToInputs(nextTransformerInput);
		}

		const transforStackOutputPostRMS3 = this.transformers[0].RMSNorm(nextTransformerInput, this.rmsGamma3, 'rms3');
		const logitScoresByInputToken = this.matMul_vocab_WEBGPU(this.tokenEmbeddings, transforStackOutputPostRMS3);
		this.logitScoresByInputTokenPostSoftmax = this.logitSoftmax_WEBGPU(logitScoresByInputToken);
	}
}
