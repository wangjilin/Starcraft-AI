import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer # Bot is myself, computer is computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY # protoss base, worker, pylon(crystal), gas station, BG and BY, stalker, VS, voidray
import random


class MyBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 65
    
    async def on_step(self, iteration): # about 165 iterations per minute when realtime=False, about 923 iterations per minute when realtime=True
        self.iteration = iteration
        # what to do every step
        await self.distribute_workers()  # already written in sc2/bot_ai.py, auto distribution workers to mine, 
        await self.build_workers() # Build worker. Not exist, need to write by ourself. Defined below
        await self.build_pylons() # Build pylons (crystal). Not exist, need to write by ourself. Defined below
        await self.build_assimilators() # Build assimilators (gas station). Not exist, need to write by ourself. Defined below
        await self.expand() # sc2.BotAI package has a function called expand_now, but it's brainless and we need to consider more situations,
                            # so we write one ourself
        await self.offensive_force_buildings() # buildings to build army.
        await self.build_offensive_force() # build army
        await self.attack() # attack
        
    async def build_workers(self): # train more worker
        if len(self.units(NEXUS))*(16+6) > len(self.units(PROBE)): # if workers in total are less than bases number * (16+6) (mine+gas)
            if len(self.units(PROBE)) <= self.MAX_WORKERS: # if workers are not more than the max worker numbers
                for nexus in self.units(NEXUS).ready.noqueue: # all already built nexus, without producing anything (probes)
                    if self.can_afford(PROBE): # if we can afford a probe
                        await self.do(nexus.train(PROBE)) #let nexus train a probe
        
    async def build_pylons(self): # build more pylons (crystal)
        if self.supply_left < 5 and not self.already_pending(PYLON): # if supply left is not enough and no building pylon in process
            nexuses = self.units(NEXUS).ready # nexus that already exist
            if nexuses.exists:   # if already have nexus
                if self.can_afford(PYLON): # if we can afford a pylon
#                    await self.build(PYLON, near=nexuses.first) # build near first nexus in the list
                    await self.build(PYLON, near= random.choice(nexuses)) # build near a random nexus in the list
                    
    async def build_assimilators(self): # build assimilators (gas station)
        for nexus in self.units(NEXUS).ready.noqueue: # all already built nexus
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus) # gas sourse closer than 15 units to each bases
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):  # if we can afford a ASSIMILATOR
                    break
                worker = self.select_build_worker(vaspene.position) # selection a worker close to the vaspenes gas
                if worker is None: # no worker, no build
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0,vaspene).exists: # if no gas station on the vaspene gas
                    await self.do(worker.build(ASSIMILATOR,vaspene)) 
                    
    async def expand(self): # expand
        if self.units(NEXUS).amount < 4 and self.can_afford(NEXUS): # expand till 4 bases
            await self.expand_now() # from sc2.BotAI package 
    
    async def offensive_force_buildings(self): # buildings to build army.
        if self.units(PYLON).ready.exists: 
            pylon = self.units(PYLON).ready.random # randomly choose a existing pylon
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE): # if have BG, no BY, build BY
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE,near = pylon)                    
            elif self.units(GATEWAY).amount < (self.iteration / self.ITERATIONS_PER_MINUTE): # build a gate way per minute
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY,near = pylon)
            # Build stargates
            if self.units(CYBERNETICSCORE).ready.exists: # if have BY
                if len(self.units(STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE)/2: # build a stargate every 2 mins
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE,near = pylon)      
            
    async def build_offensive_force(self): # train army (stalker)
        for gw in self.units(GATEWAY).ready.noqueue: # for each gateway
            if not self.units(STALKER).amount > self.units(VOIDRAY).amount: # only build staler when it's less than voidray, 
                                                                            # otherwise all the resourse will be used to build stalker 
                if self.can_afford(STALKER) and self.supply_left >= 2: # if can afford a stalker and supply is enough
                    await self.do(gw.train(STALKER)) #let gateway train a stalker
        
        for sg in self.units(STARGATE).ready.noqueue: # for each gateway
            if self.can_afford(VOIDRAY) and self.supply_left >= 2: # if can afford a stalker and supply is enough
                await self.do(sg.train(VOIDRAY)) #let gateway train a stalker
                
    def find_target(self,state):
        if self.known_enemy_units: # if already know enemy unit, return a random one of them
            random.choice(self.known_enemy_units)
        elif self.known_enemy_structures: # if already know enemy strcture, return a random one of them
            random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0] # else, attack on enemy base
                
    async def attack(self): # attack!!!
        # dict = {UNIT: [n to fight, n to defend]}
        aggressive_units = {STALKER: [15, 3],
                            VOIDRAY: [8, 3]}
        '''
        if self.units(STALKER).amount > 15: # if we have enough enemy, attack their base
             for s in self.units(STALKER).idle: # 'idle' for stalkers that are NOT currently doing anything!!!!!!!
                 await self.do(s.attack(self.find_target(self.state))) # find enemy and attack, state means the current game state   
        elif self.units(STALKER).amount > 3: 
            if self.known_enemy_units: # if there are known (already seen) enemy units
                for s in self.units(STALKER).idle: # 'idle' for stalkers that are NOT currently doing anything!!!!!!!
                    await self.do(s.attack(random.choice(self.known_enemy_units))) # let them attack a random known enemy unit
        '''
        for UNIT in aggressive_units:
            # if we have enough army to chase and attack the enemy, attack them
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))
            # if we only have enough army for defense, then defense to the attacking enemy
            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))
        
        
run_game(maps.get("AbyssalReefLE"), [ # choose a map
    Bot(Race.Protoss, MyBot()), # ourself, race and what we do
    Computer(Race.Terran, Difficulty.Hard)],  # compuer race and difficulty
    realtime=False) # realtime = False means super fast, True means real gameplay time
    

#Also, we can let two robot fight on each other
#for example, import an other robot from folder python-sc2-master
#
#For some reason, fairly often when I put two of my own bots against each other, I just get black screens when the game launches and nothing happens. 
#To fix this, I go to C:\Users\H\Documents\StarCraft II and delete variables.txt
#
#from examples.terran.proxy_rax import ProxyRaxBot
#
#run_game(maps.get("AbyssalReefLE"), [
#    Bot(Race.Protoss, MyBot()),
#    Bot(Race.Terran, ProxyRaxBot()),
#    ], realtime=False)
