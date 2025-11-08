// ==================== NEURAL NETWORK ====================
class SimpleQNetwork {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;

        this.hiddenSize = 128;
        this.weights1 = this.randomMatrix(stateSize, this.hiddenSize);
        this.bias1 = this.randomArray(this.hiddenSize);
        this.weights2 = this.randomMatrix(this.hiddenSize, this.hiddenSize);
        this.bias2 = this.randomArray(this.hiddenSize);
        this.weights3 = this.randomMatrix(this.hiddenSize, actionSize);
        this.bias3 = this.randomArray(actionSize);
    }

    randomMatrix(rows, cols) {
        const scale = Math.sqrt(2.0 / rows); // He initialization
        return Array(rows).fill(0).map(() =>
            Array(cols).fill(0).map(() => (Math.random() - 0.5) * 2 * scale)
        );
    }

    randomArray(size) {
        return Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.01);
    }

    relu(x) {
        return Math.max(0, x);
    }

    forward(state) {
        // First hidden layer
        const hidden1 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias1[i];
            for (let j = 0; j < this.stateSize; j++) {
                sum += state[j] * this.weights1[j][i];
            }
            return this.relu(sum);
        });

        // Second hidden layer
        const hidden2 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden1[j] * this.weights2[j][i];
            }
            return this.relu(sum);
        });

        // Output layer
        const output = Array(this.actionSize).fill(0).map((_, i) => {
            let sum = this.bias3[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden2[j] * this.weights3[j][i];
            }
            return sum;
        });

        return output;
    }

    update(state, action, target, learningRate) {
        // Forward pass
        const hidden1 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias1[i];
            for (let j = 0; j < this.stateSize; j++) {
                sum += state[j] * this.weights1[j][i];
            }
            return this.relu(sum);
        });

        const hidden2 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden1[j] * this.weights2[j][i];
            }
            return this.relu(sum);
        });

        const output = this.forward(state);

        // Compute error
        const outputError = output[action] - target;

        // Update output layer (layer 3)
        for (let i = 0; i < this.hiddenSize; i++) {
            this.weights3[i][action] -= learningRate * outputError * hidden2[i];
        }
        this.bias3[action] -= learningRate * outputError;

        // Backpropagate to hidden layers (simplified)
        for (let i = 0; i < this.hiddenSize; i++) {
            const hidden2Error = outputError * this.weights3[i][action];
            if (hidden2[i] > 0) {
                for (let j = 0; j < this.hiddenSize; j++) {
                    this.weights2[j][i] -= learningRate * hidden2Error * hidden1[j] * 0.1;
                }
            }
        }
    }

    copy() {
        const newNetwork = new SimpleQNetwork(this.stateSize, this.actionSize);
        newNetwork.weights1 = this.weights1.map(row => [...row]);
        newNetwork.bias1 = [...this.bias1];
        newNetwork.weights2 = this.weights2.map(row => [...row]);
        newNetwork.bias2 = [...this.bias2];
        newNetwork.weights3 = this.weights3.map(row => [...row]);
        newNetwork.bias3 = [...this.bias3];
        return newNetwork;
    }

    save() {
        return {
            stateSize: this.stateSize,
            actionSize: this.actionSize,
            hiddenSize: this.hiddenSize,
            weights1: this.weights1,
            bias1: this.bias1,
            weights2: this.weights2,
            bias2: this.bias2,
            weights3: this.weights3,
            bias3: this.bias3
        };
    }

    load(data) {
        this.stateSize = data.stateSize;
        this.actionSize = data.actionSize;
        this.hiddenSize = data.hiddenSize;
        this.weights1 = data.weights1;
        this.bias1 = data.bias1;
        this.weights2 = data.weights2;
        this.bias2 = data.bias2;
        this.weights3 = data.weights3;
        this.bias3 = data.bias3;
    }

    static fromData(data) {
        const network = new SimpleQNetwork(data.stateSize, data.actionSize);
        network.load(data);
        return network;
    }
}

module.exports = SimpleQNetwork;