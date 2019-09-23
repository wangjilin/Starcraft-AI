import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer # Bot is myself, computer is computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
# protoss base, worker, pylon(crystal), gas station, BG and BY, stalker, VS, voidray
# ob, VR
import random
import cv2
import numpy as np
import time
import os
import keras
import math

os.chdir("E:/ML_programs/Starcraft2_AI/Python AI in StarCraft II/")


HEADLESS = False # not show opencv circles

class MyBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.use_model = use_model
        self.title = title
        
        self.train_data = []
        # already sent scouting units id and location {Unit_tag: location}
        self.scouts_and_spots = {}
        # all the new choices we are going to train
        self.choices = {0: self.build_scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        5: self.build_worker,
                        6: self.build_assimilator,
                        7: self.build_stargate,
                        8: self.build_pylon,
                        9: self.defend_nexus,
                        10: self.attack_known_enemy_unit,
                        11: self.attack_known_enemy_structure,
                        12: self.expand,
                        13: self.do_nothing,
                        }
        
        self.use_model = use_model
        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("model_we_use/BasicCNN-30-epochs-0.0001-LR-4.2")
        
    def on_end(self,game_result):
#        print('---on_end called---')
#        print(game_result)
#        
        if game_result == Result.Victory: # if our bot win, save the train_data
            np.save('train_data/{}.npy'.format(str(int(time.time()))),np.array(self.train_data))
            cv2.destroyAllWindows()        
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
    async def on_step(self, iteration): # about 165 iterations per minute when realtime=False, about 923 iterations per minute when realtime=True
#        self.iteration = iteration
        # self.time is the real game time in sec!!!
        # what to do every step
        print(self.time/60)
        await self.build_scout()
        await self.distribute_workers()  # already written in sc2/bot_ai.py, auto distribution workers to mine, 
        await self.scout() # check what the enemy is doing, scout first then we can know our following steps
        '''
        await self.build_workers() # Build worker. Not exist, need to write by ourself. Defined below
        await self.build_pylons() # Build pylons (crystal). Not exist, need to write by ourself. Defined below
        await self.build_assimilators() # Build assimilators (gas station). Not exist, need to write by ourself. Defined below
        await self.expand() # sc2.BotAI package has a function called expand_now, but it's brainless and we need to consider more situations,
                            # so we write one ourself
        await self.offensive_force_buildings() # buildings to build army.
        await self.build_offensive_force() # build army
        await self.attack() # attack
        '''
        await self.intel() # deep learning part
        
        await self.do_something()
    
    async def scout(self): # send ob and check what the enemy is doing
        '''
        if len(self.units(OBSERVER)) > 0: # if we have ob, send it
            scout = self.units(OBSERVER)[0]
            if scout.is_idle: 
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location) # send it a random location near enemy base
                print(move_to)
                await self.do(scout.move(move_to))

        else: # if no ob, build one
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER)) 
        '''
        # {DISTANCE_TO_ENEMY_START:EXPANSIONLOC}
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            #print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el 
        # all the expansion location in a closest order to the enemy first base
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)  
        
        existing_ids = [unit.tag for unit in self.units] # the ID of all our own units
#        print('existing_ids: ',existing_ids)
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]
        # If no VR, send probe to scout
        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15
            
        assign_scout = True
        
        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False
        # assign scout
        if assign_scout:
            if len(self.units(unit_type).idle) > 0: # if there is idle ob/probe
                for obs in self.units(unit_type).idle[:unit_limit]: 
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = self.expand_dis_dir[dist] #next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]
                                # if there is no unit scouting the selected location 
                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            # select a probe not scouting
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    
                                    await self.do(obs.move(location))
                                    # add the scouting unit and its scouting location to the list
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass
        # if is a probe, let it walk around
        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
#                if obs in [probe for probe in self.units(PROBE)]:
                if obs in self.units(PROBE):
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))
        
        
    def random_location_variance(self, enemy_start_location): # 
        x = enemy_start_location[0]
        y = enemy_start_location[1]
        '''
        x += ((random.randrange(-20, 20))/100) * self.game_info.map_size[0]
        y += ((random.randrange(-20, 20))/100) * self.game_info.map_size[1]
        '''
        x += random.randrange(-5,5)
        y += random.randrange(-5,5)
        
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y))) 
        # !!! use point2 for position !!! see position.py for details. class Point2(Pointlike), class Pointlike(tuple)
        return go_to
    
    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            print(len(self.units(OBSERVER)), self.time/60/3)
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break                
    
    
    async def intel(self): 
        # go in python-sc2 wiki for more functions and variables, such as self.game_info.map_size
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8) # image is width by height, 3 for color,  
                                                                                                    # data type uint8

        # UNIT: [SIZE, (BGR COLOR)]
        '''
        draw_dict = {
                     NEXUS: [15, (0, 255, 0)], # circle size , color
                     PYLON: [3, (20, 235, 0)],
                     PROBE: [1, (55, 200, 0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     GATEWAY: [3, (200, 100, 0)],
                     CYBERNETICSCORE: [3, (150, 150, 0)],
                     STARGATE: [5, (255, 0, 0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)],

                     VOIDRAY: [3, (255, 100, 0)],
                     #OBSERVER: [3, (255, 255, 255)],
                    }
        '''
        # our current units
        for unit in self.units().ready:
            pos = unit.position
            # now circle size is propotional to the unit real size, all white
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))
        # known enemy units
        for unit in self.known_enemy_units:
            pos = unit.position # in color grey
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))
            
        '''
        for unit_type in draw_dict: # for each type of elements we are going to draw circle
            for unit in self.units(unit_type).ready: 
                pos = unit.position
                #draw a circle, at unit location, size draw_dict[unit_type][0], color draw_dict[unit_type][1], linewidth -1
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1) 
        
        main_base_names = ['nexus', 'commandcenter', 'orbitalcommand', 'planetaryfortress', 'hatchery']
        # enemy buildings we want a LARGE circle (protoss, human, zerg base)
        for enemy_building in self.known_enemy_structures: # others use small circle
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready: # show our OB location
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)
        '''    
        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0
    
            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0
    
            population_ratio = self.supply_left / self.supply_cap # supply_cap : Current supply cap limited by bases, supply depots
            if population_ratio > 1.0:
                population_ratio = 1.0
    
            plausible_supply = self.supply_cap / 200.0
    
            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0
    
            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            print(str(e))
            
            
        
        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY) # plot in gray
        self.flipped = cv2.flip(grayed, 0)
        
        if not HEADLESS: # if we want wo plot opcv fig
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)
        
        
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            print(len(self.units(OBSERVER)), self.time/60/3)
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break

    async def build_worker(self):
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PROBE):
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon)

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                await self.do(random.choice(stargates).train(VOIDRAY))

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def build_stargate(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

    async def build_pylon(self):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):# position.towards(self.game_info.map_center, 5), so it does not be built in the mine area
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def expand(self):
        try:
            if self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.time/60 + wait

    async def defend_nexus(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))

    async def attack_known_enemy_structure(self):
        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures)
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))

    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))

    async def do_something(self):

        if self.time/60 > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                worker_weight = 8
                zealot_weight = 3
                voidray_weight = 20
                stalker_weight = 8
                pylon_weight = 5
                stargate_weight = 5
                gateway_weight = 3
                # make different choice with weight probability
                choice_weights = 1*[0]+zealot_weight*[1]+gateway_weight*[2]+voidray_weight*[3]+stalker_weight*[4]+worker_weight*[5]+1*[6]+stargate_weight*[7]+pylon_weight*[8]+1*[9]+1*[10]+1*[11]+1*[12]+1*[13]
                choice = random.choice(choice_weights)

            try:
                await self.choices[choice]() # exclusive one of the methods in the list
            except Exception as e:
                print(str(e))
            y = np.zeros(14)
            y[choice] = 1
            self.train_data.append([y, self.flipped]) 
        
run_game(maps.get("AbyssalReefLE"), [ # choose a map
    Bot(Race.Protoss, MyBot(use_model=False, title=1)), # ourself, race and what we do; use_model=True is using the trained model to choose what to do
    Computer(Race.Terran, Difficulty.Medium)],  # compuer race and difficulty
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
