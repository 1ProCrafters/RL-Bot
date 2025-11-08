const fs = require('fs');
const path = require('path');

const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const { GoalNear, GoalBlock, GoalXZ, GoalY, GoalInvert, GoalFollow } = goals;

const { SERVER_CONFIG, TRAINING_CONFIG, ARENA_CONFIG } = require('./config');
const RLAgent = require('./RLAgent');
const OperatorBot = require('./OperatorBot');
const PvPBot = require('./PvPBot');

// ==================== TRAINING SYSTEM ====================
class TrainingSystem {
    constructor(loadModel = false) {
        this.operators = [];
        this.agent = loadModel ? this.loadAgent() : new RLAgent(TRAINING_CONFIG);
        this.bots = [];
        this.episode = 0;
        this.totalEpisodes = 0;
        this.startTime = Date.now();
        this.episodeRewards = [];
        this.episodeKills = [];
        this.episodeLengths = [];
    }

    loadAgent() {
        const modelPath = TRAINING_CONFIG.modelPath;
        if (fs.existsSync(modelPath)) {
            try {
                console.log(`Loading model from ${modelPath}...`);
                const data = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
                const agent = RLAgent.fromData(data.agent, TRAINING_CONFIG);
                this.totalEpisodes = data.totalEpisodes || 0;
                this.episodeRewards = data.episodeRewards || [];
                this.episodeKills = data.episodeKills || [];
                this.episodeLengths = data.episodeLengths || [];
                console.log(`✅ Model loaded! Resuming from episode ${this.totalEpisodes}`);
                console.log(`Current epsilon: ${agent.epsilon.toFixed(4)}`);
                return agent;
            } catch (error) {
                console.error('❌ Error loading model:', error);
                console.log('Starting fresh instead.');
                return new RLAgent(TRAINING_CONFIG);
            }
        } else {
            console.log('No saved model found. Starting fresh.');
            return new RLAgent(TRAINING_CONFIG);
        }
    }

    saveAgent() {
        try {
            const modelPath = TRAINING_CONFIG.modelPath;
            const dir = path.dirname(modelPath);

            // Create directory if it doesn't exist
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`Created directory: ${dir}`);
            }

            const data = {
                agent: this.agent.save(),
                totalEpisodes: this.totalEpisodes,
                episodeRewards: this.episodeRewards.slice(-1000),
                episodeKills: this.episodeKills.slice(-1000),
                episodeLengths: this.episodeLengths.slice(-1000),
                timestamp: new Date().toISOString(),
                config: TRAINING_CONFIG
            };

            fs.writeFileSync(modelPath, JSON.stringify(data, null, 2));
            console.log(`💾 Model saved to ${modelPath} (Episode ${this.totalEpisodes}, ε=${this.agent.epsilon.toFixed(4)})`);

            // Save backup periodically
            if (this.totalEpisodes % (TRAINING_CONFIG.saveInterval * 10) === 0) {
                const backupPath = modelPath.replace('.json', `_ep${this.totalEpisodes}.json`);
                fs.writeFileSync(backupPath, JSON.stringify(data, null, 2));
                console.log(`📦 Backup saved: ${backupPath}`);
            }
        } catch (error) {
            console.error('❌ Error saving model:', error);
        }
    }

    async initialize() {
        console.log('Initializing training system...');

        // Connect operators
        for (let i = 0; i < TRAINING_CONFIG.numOperators; i++) {
            const operator = new OperatorBot();
            // We need to give each operator a unique name
            operator.bot = mineflayer.createBot({
                ...SERVER_CONFIG,
                username: `Operator${i + 1}`
            });
            await new Promise(resolve => operator.bot.once('spawn', resolve));
            console.log(`Operator${i + 1} connected!`);
            this.operators.push(operator);
        }

        // Setup arenas, distributing the work
        await this.setupArenas();
        await new Promise(resolve => setTimeout(resolve, 1000));

        if (TRAINING_CONFIG.numBots % TRAINING_CONFIG.botsPerArena !== 0) {
            throw new Error('`numBots` must be a multiple of `botsPerArena`.');
        }

        // Create and connect bots
        for (let i = 0; i < TRAINING_CONFIG.numBots; i++) {
            const arenaIndex = Math.floor(i / TRAINING_CONFIG.botsPerArena);
            const assignedOperator = this.operators[arenaIndex % this.operators.length];
            const bot = new PvPBot(`Fighter${i + 1}`, TRAINING_CONFIG, this.agent, arenaIndex);
            bot.setOperator(assignedOperator);
            await bot.connect();
            await new Promise(resolve => setTimeout(resolve, TRAINING_CONFIG.connectDelay));
            this.bots.push(bot);
        }

        // Set opponents within each arena
        for (let i = 0; i < this.bots.length; i += 2) {
            this.bots[i].setOpponent(this.bots[i + 1]);
            this.bots[i + 1].setOpponent(this.bots[i]);
        }

        console.log('Training system initialized!');
    }

    async setupArenas() {
        const numArenas = Math.ceil(TRAINING_CONFIG.numBots / TRAINING_CONFIG.botsPerArena);
        console.log(`Setting up ${numArenas} arena(s) using ${this.operators.length} operators...`);

        const setupPromises = [];
        for (let i = 0; i < numArenas; i++) {
            const operator = this.operators[i % this.operators.length];
            setupPromises.push(this.setupSingleArena(operator, i));
        }
        await Promise.all(setupPromises);

        // Use the primary operator for global settings
        const primaryOperator = this.operators[0];
        await primaryOperator.executeCommand('/time set day');
        await primaryOperator.executeCommand('/weather clear');
        await primaryOperator.executeCommand('/gamerule doDaylightCycle false');
        await primaryOperator.executeCommand('/gamerule doMobSpawning false');
        await primaryOperator.executeCommand('/gamerule playersSleepingPercentage 0');

        console.log('Arena setup complete!');
    }

    async setupSingleArena(operator, arenaIndex) {
        const size = ARENA_CONFIG.size;
        const arenaX = ARENA_CONFIG.center.x + arenaIndex * TRAINING_CONFIG.arenaSpacing;
        const { y, z } = ARENA_CONFIG.center;

        console.log(`  - ${operator.bot.username} creating arena ${arenaIndex + 1} at x=${arenaX}`);

        // Run all setup commands for a single arena concurrently
        const commands = [
            `/fill ${arenaX - size} ${y - 5} ${z - size} ${arenaX + size} ${y + 20} ${z + size} air`, // Clear area
            `/fill ${arenaX - size} ${y - 1} ${z - size} ${arenaX + size} ${y - 1} ${z + size} stone`, // Floor
            `/fill ${arenaX - size} ${y} ${z - size} ${arenaX - size} ${y + 5} ${z + size} barrier`, // Wall 1
            `/fill ${arenaX + size} ${y} ${z - size} ${arenaX + size} ${y + 5} ${z + size} barrier`, // Wall 2
            `/fill ${arenaX - size} ${y} ${z - size} ${arenaX + size} ${y + 5} ${z - size} barrier`, // Wall 3
            `/fill ${arenaX - size} ${y} ${z + size} ${arenaX + size} ${y + 5} ${z + size} barrier`  // Wall 4
        ];

        await Promise.all(commands.map(cmd => operator.executeCommand(cmd)));
    }

    async runEpisode() {
        this.episode++;
        this.totalEpisodes++;
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🎮 Episode ${this.totalEpisodes} (Session: ${this.episode}/${TRAINING_CONFIG.episodesPerSession})`);
        console.log(`${'='.repeat(60)}`);

        // Reset bots
        for (const bot of this.bots) {
            bot.resetEpisode();
            await bot.operator.healPlayer(bot.name);
            await bot.operator.giveEquipment(bot.name);
            const spawnX = (Math.random() - 0.5) * (ARENA_CONFIG.size - 2);
            const spawnZ = (Math.random() - 0.5) * (ARENA_CONFIG.size - 2); // This line was correct, but the next one was missing the arenaIndex
            await bot.operator.teleportPlayer(bot.name, spawnX, ARENA_CONFIG.respawnY, spawnZ, bot.arenaIndex);
        }

        await new Promise(resolve => setTimeout(resolve, 3000));

        // Run episode
        const episodeStartTime = Date.now();
        let steps = 0;
        const activeArenas = new Set(this.bots.map(b => b.arenaIndex));

        while (steps < TRAINING_CONFIG.maxStepsPerEpisode) {
            for (const bot of this.bots) {
                // Only step if the bot's arena is still active
                if (activeArenas.has(bot.arenaIndex)) {
                    await bot.step();
                }
            }
            await new Promise(resolve => setTimeout(resolve, 50));
            steps++;

            // Check for finished fights in each arena
            const finishedArenas = new Set();
            for (const arenaIndex of activeArenas) {
                const botsInArena = this.bots.filter(b => b.arenaIndex === arenaIndex);
                if (botsInArena.some(b => b.bot && b.bot.health <= 0)) {
                    finishedArenas.add(arenaIndex);
                }
            }

            for (const arenaIndex of finishedArenas) {
                activeArenas.delete(arenaIndex);
            }

            if (activeArenas.size === 0) break;
        }

        const episodeDuration = (Date.now() - episodeStartTime) / 1000;

        // Collect episode stats
        const totalReward = this.bots.reduce((sum, bot) => sum + bot.episodeReward, 0) / this.bots.length;
        const totalKills = this.bots.reduce((sum, bot) => sum + bot.kills, 0);

        this.episodeRewards.push(totalReward);
        this.episodeKills.push(totalKills);
        this.episodeLengths.push(steps);

        // Update target network periodically
        if (this.totalEpisodes % TRAINING_CONFIG.targetNetworkUpdateInterval === 0) {
            this.agent.updateTargetNetwork();
            console.log('🎯 Target network updated');
        }

        // Save model periodically
        if (this.totalEpisodes % TRAINING_CONFIG.saveInterval === 0) {
            this.saveAgent();
        }

        // Print stats
        if (this.episode % TRAINING_CONFIG.logInterval === 0) {
            this.printDetailedStats(episodeDuration, steps);
        } else {
            console.log(`⏱️  Duration: ${episodeDuration.toFixed(1)}s | Steps: ${steps} | Avg Reward: ${totalReward.toFixed(1)} | ε: ${this.agent.epsilon.toFixed(4)}`);
        }
    }

    printDetailedStats(lastEpisodeDuration, steps) {
        console.log('\n' + '='.repeat(60));
        console.log('📊 DETAILED STATISTICS');
        console.log('='.repeat(60));

        const totalTime = (Date.now() - this.startTime) / 1000;
        const avgTime = totalTime / this.episode;

        console.log(`\n⏱️  Time Stats:`);
        console.log(`   Total Training Time: ${(totalTime / 60).toFixed(2)} minutes`);
        console.log(`   Avg Episode Duration: ${avgTime.toFixed(2)}s`);
        console.log(`   Last Episode: ${lastEpisodeDuration.toFixed(2)}s (${steps} steps)`);

        console.log(`\n🤖 Agent Stats:`);
        console.log(`   Total Episodes: ${this.totalEpisodes}`);
        console.log(`   Memory Size: ${this.agent.memory.length} / ${this.agent.maxMemorySize}`);
        console.log(`   Exploration (ε): ${this.agent.epsilon.toFixed(4)}`);
        console.log(`   Learning Rate: ${TRAINING_CONFIG.learningRate}`);
        console.log(`   Batch Size: ${TRAINING_CONFIG.batchSize}`);

        // Recent performance (last 10 episodes)
        const recent = Math.min(10, this.episodeRewards.length);
        const recentRewards = this.episodeRewards.slice(-recent);
        const recentKills = this.episodeKills.slice(-recent);
        const recentLengths = this.episodeLengths.slice(-recent);

        const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
        const avgKills = recentKills.reduce((a, b) => a + b, 0) / recentKills.length;
        const avgLength = recentLengths.reduce((a, b) => a + b, 0) / recentLengths.length;

        console.log(`\n📈 Recent Performance (last ${recent} episodes):`);
        console.log(`   Avg Reward: ${avgReward.toFixed(2)}`);
        console.log(`   Avg Kills per Episode: ${avgKills.toFixed(2)}`);
        console.log(`   Avg Episode Length: ${avgLength.toFixed(0)} steps`);

        console.log(`\n⚔️  Bot Performance:`);
        for (const bot of this.bots) {
            const stats = bot.getStats();
            console.log(`   ${stats.name}:`);
            console.log(`      K/D Ratio: ${stats.kd} (${stats.kills}K / ${stats.deaths}D)`);
            console.log(`      Hits: ${stats.hits}`);
            console.log(`      Damage: ${stats.damageDealt} dealt / ${stats.damageTaken} taken`);
            console.log(`      Last Reward: ${stats.lastReward}`);
        }

        // All-time best
        if (this.episodeRewards.length > 0) {
            const bestReward = Math.max(...this.episodeRewards);
            const bestKills = Math.max(...this.episodeKills);
            const shortestWin = Math.min(...this.episodeLengths.filter((_, i) => this.episodeKills[i] > 0));

            console.log(`\n🏆 All-Time Records:`);
            console.log(`   Best Reward: ${bestReward.toFixed(2)}`);
            console.log(`   Most Kills: ${bestKills}`);
            if (shortestWin !== Infinity) {
                console.log(`   Fastest Kill: ${shortestWin} steps`);
            }
        }

        console.log('\n' + '='.repeat(60) + '\n');
    }

    async train() {
        await this.initialize();

        console.log('\n🚀 Starting training...');
        console.log(`Training for ${TRAINING_CONFIG.episodesPerSession} episodes`);
        console.log(`Saving every ${TRAINING_CONFIG.saveInterval} episodes to ${TRAINING_CONFIG.modelPath}\n`);

        for (let i = 0; i < TRAINING_CONFIG.episodesPerSession; i++) {
            await this.runEpisode();
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Final save
        this.saveAgent();

        console.log('\n' + '='.repeat(60));
        console.log('✅ TRAINING COMPLETE');
        console.log('='.repeat(60));
        console.log(`Total Episodes: ${this.totalEpisodes}`);
        console.log(`Final ε: ${this.agent.epsilon.toFixed(4)}`);
        console.log(`Model saved to: ${TRAINING_CONFIG.modelPath}`);

        console.log('\n⚔️  Final Bot Stats:');
        for (const bot of this.bots) {
            const stats = bot.getStats();
            console.log(`${stats.name}: ${stats.kills}K / ${stats.deaths}D (K/D: ${stats.kd})`);
        }

        if (this.episodeRewards.length >= 10) {
            const avgReward = this.episodeRewards.slice(-10).reduce((a, b) => a + b, 0) / 10;
            console.log(`\nFinal 10-episode avg reward: ${avgReward.toFixed(2)}`);
        }
        console.log('='.repeat(60) + '\n');
    }

    cleanup() {
        console.log('Cleaning up...');
        for (const bot of this.bots) {
            if (bot.bot) {
                try {
                    bot.bot.quit();
                } catch (e) { }
            }
        }
        for (const operator of this.operators) {
            operator.disconnect();
        }
    }
}

module.exports = TrainingSystem;