const { SERVER_CONFIG, TRAINING_CONFIG, ARENA_CONFIG } = require('./config');

// ==================== OPERATOR BOT ====================
class OperatorBot {
    constructor() {
        this.bot = null;
    }

    async connect() {
        this.bot = mineflayer.createBot({
            ...SERVER_CONFIG,
            username: 'Operator'
        });

        return new Promise((resolve, reject) => {
            this.bot.once('spawn', () => {
                console.log('Operator connected!');
                resolve();
            });

            this.bot.once('error', reject);
            this.bot.once('kicked', (reason) => reject(new Error(reason)));
        });
    }

    async setupArena() {
        const numArenas = Math.ceil(TRAINING_CONFIG.numBots / TRAINING_CONFIG.botsPerArena);
        console.log(`Setting up ${numArenas} arena(s)...`);

        const size = ARENA_CONFIG.size;

        for (let i = 0; i < numArenas; i++) {
            const arenaX = ARENA_CONFIG.center.x + i * TRAINING_CONFIG.arenaSpacing;
            const { y, z } = ARENA_CONFIG.center;

            console.log(`  - Creating arena ${i + 1} at x=${arenaX}`);

            // Clear area
            await this.executeCommand(`/fill ${arenaX - size} ${y - 5} ${z - size} ${arenaX + size} ${y + 20} ${z + size} air`);
            await new Promise(resolve => setTimeout(resolve, 100));

            // Create floor
            await this.executeCommand(`/fill ${arenaX - size} ${y - 1} ${z - size} ${arenaX + size} ${y - 1} ${z + size} stone`);

            // Create walls
            await this.executeCommand(`/fill ${arenaX - size} ${y} ${z - size} ${arenaX - size} ${y + 5} ${z + size} barrier`);
            await this.executeCommand(`/fill ${arenaX + size} ${y} ${z - size} ${arenaX + size} ${y + 5} ${z + size} barrier`);
            await this.executeCommand(`/fill ${arenaX - size} ${y} ${z - size} ${arenaX + size} ${y + 5} ${z - size} barrier`);
            await this.executeCommand(`/fill ${arenaX - size} ${y} ${z + size} ${arenaX + size} ${y + 5} ${z + size} barrier`);
        }

        // Set time and weather
        await this.executeCommand('/time set day');
        await this.executeCommand('/weather clear');
        await this.executeCommand('/gamerule doDaylightCycle false');
        await this.executeCommand('/gamerule doMobSpawning false');
        await this.executeCommand('/gamerule playersSleepingPercentage 0');

        console.log('Arena setup complete!');
    }

    async giveEquipment(playerName) {
        await this.executeCommand(`/clear ${playerName}`);
        await this.executeCommand(`/give ${playerName} diamond_sword 1`);
        await this.executeCommand(`/give ${playerName} golden_apple 64`);
        await this.executeCommand(`/give ${playerName} diamond_helmet{Enchantments:[{id:"protection",lvl:4}]} 1`);
        await this.executeCommand(`/give ${playerName} diamond_chestplate{Enchantments:[{id:"protection",lvl:4}]} 1`);
        await this.executeCommand(`/give ${playerName} diamond_leggings{Enchantments:[{id:"protection",lvl:4}]} 1`);
        await this.executeCommand(`/give ${playerName} diamond_boots{Enchantments:[{id:"protection",lvl:4}]} 1`);
    }

    async healPlayer(playerName) {
        await this.executeCommand(`/effect clear ${playerName}`);
        await this.executeCommand(`/effect give ${playerName} instant_health 1 10`);
        await this.executeCommand(`/effect give ${playerName} saturation 1 10`);
    }

    async teleportPlayer(playerName, x, y, z, arenaIndex = 0) {
        const arenaXOffset = arenaIndex * TRAINING_CONFIG.arenaSpacing;
        const finalX = ARENA_CONFIG.center.x + arenaXOffset + x;
        const finalY = y;
        const finalZ = ARENA_CONFIG.center.z + z;

        await this.executeCommand(`/tp ${playerName} ${finalX} ${finalY} ${finalZ}`);
    }

    async executeCommand(command) {
        return new Promise((resolve) => {
            this.bot.chat(command);
            setTimeout(resolve, 50);
        });
    }

    disconnect() {
        if (this.bot) {
            this.bot.quit();
        }
    }
}

module.exports = OperatorBot;