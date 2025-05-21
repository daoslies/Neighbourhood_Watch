import torch
import random
import matplotlib.pyplot as plt
import numpy as np

class Environment:
  def __init__(self):

    self.building_schema = {0 : {'building' : 'Holiday',
                                    'cost' : {'ore' : 0, 'wood' : 0, 'manpower' : 0},
                                    'construct' : 0},
                            1 : { 'building' : 'House',
                                    'cost' : {'ore' : 4, 'wood' : 8, 'manpower' : 2},
                                    'construct' : 1},
                            2 : { 'building' : 'Farm',
                                    'cost' : {'ore' : 3, 'wood' : 6, 'manpower' : 3},
                                    'construct' : 2},
                            3 : { 'building' : 'Mine',
                                    'cost' : {'ore' : 30, 'wood' : 7, 'manpower' : 10},
                                    'construct' : 3},
                            4 : { 'building' : 'LumberMill',
                                    'cost' : {'ore' : 5, 'wood' : 20, 'manpower' : 10},
                                    'construct' : 4},
                            5 : { 'building' : 'Latrine',
                                    'cost' : {'ore' : 2, 'wood' : 4, 'manpower' : 1},
                                    'construct' : 5,
                                    'full': 30}
                   }
    self.reset()
    pass
  
  def init_shire(self):

    shire_tense = torch.randint(10, [3,10,10])
    buildings = torch.zeros([1,10,10])
    shire_tense = torch.concatenate((shire_tense,buildings), axis=0)

    shire = {'farming' : shire_tense[0], 'mining' : shire_tense[1],\
             'woodcutting' : shire_tense[2], 'buildings' : shire_tense[3]}
    return shire
  
  def reset(self):
    # Reset the world
    self.timestep = 0
    #self.max_timestep = 500
    self.end = False
    self.print = False

    self.food = 400
    self.ore = 50
    self.wood = 100
    self.people = 20
    self.poop = 0
    self.appetite = 3
    
    self.reward = 0

    self.food_list = []
    self.food_eaten_list = []
    self.food_produced_list = []
    self.ore_list = []
    self.wood_list = []
    self.people_list = []
    self.poop_list = []
    self.empty_plots_list = []
    self.houses_list = []
    self.farms_list = []
    self.mines_list = []
    self.lumber_mills_list = []
    self.latrines_list = []
    self.baby_list = []

    self.shire = self.init_shire()
    
    self.resources = {'food': self.food,
                       'ore': self.ore,
                       'wood': self.wood,
                       'people': self.people,
                       'poop': self.poop}

    land_tensors = torch.tensor(np.array(list(self.shire.values()))).to(torch.float32)

    resource_tensors = torch.tensor(np.array(list(self.resources.values()))).to(torch.float32).reshape((1,5))

    return land_tensors, resource_tensors

  def step(self, council_decision):
      
    self.reward = 0
    self.available_men = int(self.people)
    self.unhoused = int(self.people) ###
    self.baby_count = 0
    self.first_food_check = self.food

    self.whatsOnTheMap()
    if self.print:
        print(f"Day {self.timestep} \nYou own {self.total_properties} properties! \n {int(self.people)} live here")

    building_plans = self.building_schema[council_decision[0]]

    self.constructionYard(building_plans, council_decision[1], council_decision[2]) 

    self.lifeAndDeath()

    self.timestep += 1
    
    self.resources = {'food': self.food,
                       'ore': self.ore,
                       'wood': self.wood,
                       'people': self.people,
                       'poop': self.poop}

    land_tensors = torch.tensor(np.array(list(self.shire.values()))).to(torch.float32)

    resource_tensors = torch.tensor(np.array(list(self.resources.values()))).to(torch.float32).reshape((1,5))

    return land_tensors, resource_tensors #reward, done, info


  # people related functions
  def lifeAndDeath(self):

    self.eatingAndExcreting()
    self.workDay()
    self.death()

    #self.babies()
    if self.print:
        if self.timestep%50 ==0:
            print(self.shire['buildings'])
            print("Food ",int(self.food))
            print("wood ",int(self.wood))
            print("Ore ",int(self.ore))
            print("\n")

    self.food_list.append(int(self.food))
    self.poop_list.append(int(self.poop))
    self.ore_list.append(int(self.ore))
    self.wood_list.append(int(self.wood))
    self.people_list.append(int(self.people))
    self.baby_list.append(int(self.baby_count))


  def eatingAndExcreting(self):

    food_eaten = ((0.05*self.people)*self.people*self.appetite)
    self.food_eaten_list.append(food_eaten)
    
    if self.food - food_eaten <= 0:
      self.food = 0
    else:
      self.food -= food_eaten
      
    self.poop += self.people
    # latrines take 50 poop

  def resourceCollection(self, x, y):
      men_scaling = 1
      if self.initial_available_men > 100:
        men_scaling = np.log(int(self.initial_available_men))

      if self.available_men > 0:
        if self.shire['buildings'][x][y]== 0:
              pass
        if self.shire['buildings'][x][y]== 1: #here
            #if self.baby_count < (self.available_men_check * 0.25):
            self.babies()

        ## Is it a farm?
        if self.shire['buildings'][x][y]== 2:
              self.food = int(self.food + int(self.shire['farming'][x][y]) * 1.5 * men_scaling)
              self.available_men = int(self.available_men - 1 * men_scaling)

        ## Is it a farm?
        if self.shire['buildings'][x][y]== 3:
              self.ore = int(self.ore + int(self.shire['mining'][x][y]) * 0.5 * men_scaling)
              self.available_men = int(self.available_men - 1 * men_scaling)

        ## Is it a farm?
        if self.shire['buildings'][x][y]== 4:
              self.wood = int(self.wood + int(self.shire['woodcutting'][x][y]) * 0.5 * men_scaling)
              self.available_men = int(self.available_men - 1 * men_scaling)

        if self.shire['buildings'][x][y] == 5:
              self.poop -= int(3 * men_scaling)
              self.available_men = int(self.available_men - 1 * men_scaling)
              if self.poop <=0:
                  self.poop = 0

  def workDay(self):
      
    self.food_produced_check = self.food
    self.available_men = int(self.available_men)
    self.initial_available_men = self.available_men

    while self.available_men > 0:
      self.available_men_check = self.available_men
      for building_coord in self.building_coordinates.tolist():

        x = [building_coord[0]][0]
        y = [building_coord[1]][0]

        self.resourceCollection(x, y)

      if self.available_men_check == self.available_men:
        break

    food_produced = self.food - self.food_produced_check
    self.food_produced_list.append(food_produced)
    
    if self.baby_count > 0:
        
        if self.food - self.first_food_check > 0:
            
            self.people += self.baby_count
            
            self.reward +=  1 * np.log(self.baby_count) #0.5
        
        
    
    self.reward += np.log10(self.people) * 0.05
    
    
    if food_produced > 0:
        self.reward += 0.0005
    
    if self.poop >= 1:
        self.reward -= 0.005 * np.log(self.poop)

    if self.properties[1] > (self.people / 20): 
        self.reward -= 0.005 * ((self.people/20) / self.properties[1])

  def death(self, deathetite=0.25):
        # death from lack of food
        if self.food == 0:
            if self.people > 0:
                the_dead = max(int(self.people * deathetite),1)
                self.people -= the_dead
                self.reward -= 0.02 * np.log10(the_dead)
        
        if self.poop >=1:
            if random.randint(0,10) == 1:
                death_from_disease = random.randint(0, min(int(self.poop), int(self.people*0.4))) # 3000, 666
                self.people = self.people - death_from_disease

        if self.people <= 0:
            self.people = 0
            print("Your people have starved! There is no one left to build the city.")
            self.reward -= 1
            self.end = True

  def babies(self, baby_growth_thresh=1, baby_probability=1):
        
        potential_baby = 0
        #if self.food - self.first_food_check >= baby_growth_thresh:
        potential_baby = random.randint(0, baby_probability)
    
        if potential_baby == 1:
            #self.people +=1
            self.baby_count +=1
            # random twins
            twins = random.randint(1,4)
            if twins == 1:
              #self.people += 1
              self.baby_count +=1

  def whatsOnTheMap(self):
      
    self.building_coordinates = self.shire['buildings'].nonzero()
    self.total_properties = len(self.building_coordinates.tolist())
    
    self.properties = torch.bincount(self.shire['buildings'].int().flatten(), minlength=6)
    self.doomsday_book = torch.bincount(self.shire['buildings'].int().flatten()).tolist()

    if len(self.doomsday_book) < len(self.building_schema.values()):
      for i in range(len(self.building_schema.values()) - len(self.doomsday_book)):
        self.doomsday_book.append(0)

    self.empty_plots_list.append(self.doomsday_book[0])
    self.houses_list.append(self.doomsday_book[1])
    self.farms_list.append(self.doomsday_book[2])
    self.mines_list.append(self.doomsday_book[3])
    self.lumber_mills_list.append(self.doomsday_book[4])
    self.latrines_list.append(self.doomsday_book[5])
    #return properties, doomsday_book # building_coordinates,


  def constructionYard(self, building_plans, x, y):
      
    cost_scaling = max(self.properties[building_plans['construct']] * 2, 0.8)
    ore_cost = building_plans['cost']['ore'] * cost_scaling
    wood_cost = building_plans['cost']['wood'] * cost_scaling
    man_cost = building_plans['cost']['manpower'] * cost_scaling
    
    if ore_cost < self.ore and wood_cost < self.wood and man_cost < self.available_men:
        self.shire['buildings'][x][y] = building_plans['construct']
        self.ore = int(self.ore) - int(ore_cost)
        self.wood = int(self.wood) - int(wood_cost)
        self.available_men = int(self.available_men) - int(man_cost)
        if self.print:
            if building_plans['building'] != 'Holiday':
                print(f"Your workers have built the {building_plans['building'].upper()} as the All Powerful Council has Demanded")
            else:
                print(f" Your Council has declared a {building_plans['building']}, \n The people of the town Rejoice and Celebrate their good fortune.")
                
        self.reward += 0.05

    else: 
        if self.print:
            print("There are not enough resources to imporove the town.")
        self.reward -= 0.005

#env = Environment()

## Reward for multiple building