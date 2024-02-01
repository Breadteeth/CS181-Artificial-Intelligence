import pygame
import copy
from sprites import *
from settings import Settings

class State:

    def __init__(self):
        self.hero = None
        self.enemies = []
        self.bullets = []
        self.eBullets = []
        self.walls = None
        self.isEnd = False
        self.score = 0

    # 将坐标离散化
    def getCoordinate(self, xy):
        return int(xy[0]/50), int(xy[1]/50)
    
    # 获取玩家合规（不碰墙）的动作
    def getActions(self, agentIndex):
        actions = []
        if (agentIndex == -1):
            x, y = self.getCoordinate(self.hero[0])
        elif (agentIndex >= len(self.enemies)):
            return actions
        else:            
            x, y = self.getCoordinate(self.enemies[agentIndex][0])
        if (0 < x <= 18) and ((Settings.MAP_ONE[y][x-1] != 1) and (Settings.MAP_ONE[y][x-1] != 2)):
            actions.append(Settings.LEFT)
        if (0 <= x < 18) and ((Settings.MAP_ONE[y][x+1] != 1) and (Settings.MAP_ONE[y][x+1] != 2)):
            actions.append(Settings.RIGHT)
        if (0 < y <= 12) and ((Settings.MAP_ONE[y-1][x] != 1) and (Settings.MAP_ONE[y-1][x] != 2)):
            actions.append(Settings.UP)
        if (0 <= y < 12) and ((Settings.MAP_ONE[y+1][x] != 1) and (Settings.MAP_ONE[y+1][x] != 2)):
            actions.append(Settings.DOWN)
        return actions
    
    # 预测子弹是否会命中目标
    def hit(self, bullet, sprite):
        x, y = self.getCoordinate(bullet[0])
        x2, y2 = self.getCoordinate(sprite[0])
        if(x != x2) and (y != y2):
            return False
        elif(x == x2) and (y == y2):
            return True
        
        elif (y == y2):
            if(x2 <= x <= x2+4) and (bullet[1] == Settings.LEFT):
                return True
            elif(x2-4 <= x <= x2) and (bullet[1] == Settings.RIGHT):
                return True
        else:
            if(y2 <= y <= y2+4) and (bullet[1] == Settings.UP):
                return True
            elif(y2 -4 <= y <= y2) and (bullet[1] == Settings.DOWN):
                return True
        '''elif (y == y2):
            if(x2 <= x <= int(sprite[0][0]+125)/50) and (bullet[1] == Settings.LEFT):
                return True
            elif(int(sprite[0][0]-125)/50 <= x <= x2) and (bullet[1] == Settings.RIGHT):
                return True
        else:
            if(y2 <= y <= int(sprite[0][1]+125)/50) and (bullet[1] == Settings.UP):
                return True
            elif(int(sprite[0][1]-125)/50 <= y <= y2) and (bullet[1] == Settings.DOWN):
                return True'''
        return False
            
    # 评判玩家动作的好坏
    def evaluate(self):
        self.score = Settings.SCORE
        x, y = self.getCoordinate(self.hero[0])

        # 尽可能解决敌人
        minDistance = 999
        for enemy in self.enemies:
            x2, y2 = self.getCoordinate(enemy[0])
            distance = abs(x - x2) + abs(y - y2)
            if(distance < minDistance):
                minDistance = distance
        if(minDistance != 999):
            self.score -= distance * 3

        # 预测是否命中敌人
        for bullet in self.bullets:
            for enemy in self.enemies:
                if self.hit(bullet, enemy):
                    # bullet.kill()
                    self.score += 100
                    break
        # 预测是否胜利，有可能会导致玩家和最后一个敌人1换1
        if len(self.enemies) == 0: 
            self.isEnd = True
            self.score += 500
        
        # 预测玩家是否会被击中
        for bullets in self.eBullets:
            for bullet in bullets:
                if self.hit(bullet, self.hero):
                    # bullet.kill()
                    # self.hero.kill()
                    self.isEnd = True
                    self.score -= 500
        
        # 不希望玩家出现在敌人正前方，容易被突袭
        for enemy in self.enemies:            
            if self.hit(enemy, self.hero):
                self.score -= 500
        return self.score
    
    # 获取下一步的状态
    def nextState(self, action, agentIndex):
        
        state = copy.deepcopy(self)
        
        if(agentIndex == -1):
            if(action == Settings.LEFT):
                state.hero[0][0] -= 50
            elif(action == Settings.RIGHT):
                state.hero[0][0] += 50
            elif(action == Settings.UP):
                state.hero[0][1] -= 50
            elif(action == Settings.DOWN):
                state.hero[0][1] += 50
        else:
            if(agentIndex >= len(state.enemies)):
                return
            enemy = state.enemies[agentIndex]
            if action == Settings.LEFT:
                enemy[0][0] -= 25
            elif action == Settings.RIGHT:
                enemy[0][0] += 25
            elif action == Settings.UP:
                enemy[0][1] -= 25
            elif action == Settings.DOWN:
                enemy[0][1] += 25
            
        if(agentIndex == len(state.enemies) - 1):
            for bullet in state.bullets:
                if bullet[1] == Settings.LEFT:
                    bullet[0][0] -= 125
                elif bullet[1] == Settings.RIGHT:
                    bullet[0][0] += 125
                elif bullet[1] == Settings.UP:
                    bullet[0][1] -= 125
                elif bullet[1] == Settings.DOWN:
                    bullet[0][1] += 125

            for bullets in state.eBullets:
                for bullet in bullets:                
                    if bullet[1] == Settings.LEFT:
                        bullet[0][0] -= 125
                    elif bullet[1] == Settings.RIGHT:
                        bullet[0][0] += 125
                    elif bullet[1] == Settings.UP:
                        bullet[0][1] -= 125
                    elif bullet[1] == Settings.DOWN:
                        bullet[0][1] += 125
                        
        if len(self.enemies) == 0: 
            self.isEnd = True
            
        for bullets in self.eBullets:
            for bullet in bullets:
                if self.hit(bullet, self.hero):
                    # bullet.kill()
                    # self.hero.kill()
                    self.isEnd = True
        ''''''
        return state
    
    # 获取未来所有状态
    def nextStates(self, agentIndex):
        if self.isEnd:
            return []
        actions = self.getActions(agentIndex)
        nextStates = [self.nextState(action, agentIndex) for action in actions]
        return nextStates
    
    '''# 获取玩家合规（不碰墙）的动作
    def getActions(self):
        actions = []
        x, y = self.getCoordinate(self.hero[0])
        if (x > 0) and ((Settings.MAP_ONE[y][x-1] != 1) and (Settings.MAP_ONE[y][x-1] != 2)):
            actions.append(Settings.LEFT)
        if (x < 18) and ((Settings.MAP_ONE[y][x+1] != 1) and (Settings.MAP_ONE[y][x+1] != 2)):
            actions.append(Settings.RIGHT)
        if (y > 0) and ((Settings.MAP_ONE[y-1][x] != 1) and (Settings.MAP_ONE[y-1][x] != 2)):
            actions.append(Settings.UP)
        if (y < 12) and ((Settings.MAP_ONE[y+1][x] != 1) and (Settings.MAP_ONE[y+1][x] != 2)):
            actions.append(Settings.DOWN)
        return actions
    
    # 考虑了方向，更精准的命中预测
    def hit(self, bullet, sprite):
        x, y = self.getCoordinate(bullet[0])
        x2, y2 = self.getCoordinate(sprite[0])
        if(x != x2) and (y != y2):
            return False
        elif(x == x2) and (y == y2):
            return True
        
        elif (y == y2):
            if(x2 <= x <= x2+4) and (bullet[1] == Settings.LEFT):
                return True
            elif(x2-4 <= x <= x2) and (bullet[1] == Settings.RIGHT):
                return True
        else:
            if(y2 <= y <= y2+4) and (bullet[1] == Settings.UP):
                return True
            elif(y2 -4 <= y <= y2) and (bullet[1] == Settings.DOWN):
                return True
        return False    
    
    # 忽略方向的粗略命中预测,适用于保守的策略
    def hit2(self, bullet, sprite):
        x, y = self.getCoordinate(bullet[0])
        x2, y2 = self.getCoordinate(sprite[0])
        if(x != x2) and (y != y2):
            return False
        elif(x == x2) and (y == y2):
            return True
        
        elif (y == y2):
            if(x2 <= x <= x2+4):
                return True
            elif(x2-4 <= x <= x2):
                return True
        else:
            if(y2 <= y <= y2+4):
                return True
            elif(y2 -4 <= y <= y2):
                return True
            
    # 评判玩家动作的好坏
    def evaluate(self):
        self.score = Settings.SCORE
        x, y = self.getCoordinate(self.hero[0])

        # 尽可能解决敌人
        minDistance = 999
        for enemy in self.enemies:
            x2, y2 = self.getCoordinate(enemy[0])
            distance = abs(x - x2) + abs(y - y2)
            if(distance < minDistance):
                minDistance = distance
        if(minDistance != 999):
            self.score -= distance * 3

        # 预测是否命中敌人
        for bullet in self.bullets:
            for enemy in self.enemies:
                if self.hit(bullet, enemy):
                    # bullet.kill()
                    self.score += 100
                    break
        # 预测是否胜利，有可能会导致玩家和最后一个敌人1换1
        if len(self.enemies) == 0: 
            self.isEnd = True
            self.score += 500
        
        # 预测玩家是否会被击中
        for bullets in self.eBullets:
            for bullet in bullets:
                if self.hit(bullet, self.hero):
                    # bullet.kill()
                    # self.hero.kill()
                    self.isEnd = True
                    self.score -= 300
        
        # 不希望玩家出现在敌人正前方，容易被突袭
        for enemy in self.enemies:            
            if self.hit(enemy, self.hero):
                self.score -= 100
        return self.score
    
    # 获取下一步的状态
    def nextState(self, action):
        
        state = copy.deepcopy(self)
        
        if(action == Settings.LEFT):
            state.hero[0][0] -= 50
        elif(action == Settings.RIGHT):
            state.hero[0][0] += 50
        elif(action == Settings.UP):
            state.hero[0][1] -= 50
        elif(action == Settings.DOWN):
            state.hero[0][1] += 50

        for enemy in state.enemies:
            if action == Settings.LEFT:
                enemy[0][0] -= 25
            elif action == Settings.RIGHT:
                enemy[0][0] += 25
            elif action == Settings.UP:
                enemy[0][1] -= 25
            elif action == Settings.DOWN:
                enemy[0][1] += 25
            
        for bullet in state.bullets:
            if bullet[1] == Settings.LEFT:
                bullet[0][0] -= 125
            elif bullet[1] == Settings.RIGHT:
                bullet[0][0] += 125
            elif bullet[1] == Settings.UP:
                bullet[0][1] -= 125
            elif bullet[1] == Settings.DOWN:
                bullet[0][1] += 125

        for bullets in state.eBullets:
            for bullet in bullets:                
                if bullet[1] == Settings.LEFT:
                    bullet[0][0] -= 125
                elif bullet[1] == Settings.RIGHT:
                    bullet[0][0] += 125
                elif bullet[1] == Settings.UP:
                    bullet[0][1] -= 125
                elif bullet[1] == Settings.DOWN:
                    bullet[0][1] += 125
                        
        if len(self.enemies) == 0: 
            self.isEnd = True
            
        for bullets in self.eBullets:
            for bullet in bullets:
                if self.hit(bullet, self.hero):
                    self.isEnd = True
        ''''''
        return state
    
    # 获取未来所有状态
    def nextStates(self):
        if self.isEnd:
            return []
        actions = self.getActions()
        nextStates = [self.nextState(action) for action in actions]
        return nextStates'''
    




class TankWar:

    def __init__(self):
        self.screen = pygame.display.set_mode(Settings.SCREEN_RECT.size)
        self.clock = pygame.time.Clock()
        self.game_still = True
        self.hero = None
        self.enemies = None
        self.enemy_bullets = None
        self.walls = None
        self.state = State()

    # 更新当前状态
    def UpdateState(self):
        self.state = State()
        self.state.hero = [[self.hero.rect.centerx, self.hero.rect.centery], self.hero.direction]  
        for bullet in self.hero.bullets:
            self.state.bullets.append([[bullet.rect.centerx, bullet.rect.centery], bullet.direction])
        i = 0
        for enemy in self.enemies:   
            self.state.enemies.append([[enemy.rect.centerx, enemy.rect.centery], enemy.direction])
            self.state.eBullets.append([])
            for bullet in enemy.bullets:
                self.state.eBullets[i].append([[bullet.rect.centerx, bullet.rect.centery], bullet.direction])
            i += 1

    @staticmethod
    def __init_game():
        """
        初始化游戏的一些设置
        :return:
        """
        pygame.init()   # 初始化pygame模块
        pygame.display.set_caption(Settings.GAME_NAME)  # 设置窗口标题
        pygame.mixer.init()    # 初始化音频模块

    def __create_sprite(self):
        self.hero = Hero(Settings.HERO_IMAGE_NAME, self.screen)
        self.enemies = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        for i in range(Settings.ENEMY_COUNT):
            direction = random.randint(0, 3)
            enemy = Enemy(Settings.ENEMY_IMAGES[direction], self.screen)
            enemy.direction = direction
            self.enemies.add(enemy)
        self.__draw_map()

    def __draw_map(self):
        """
        绘制地图
        :return:
        """
        for y in range(len(Settings.MAP_ONE)):
            for x in range(len(Settings.MAP_ONE[y])):
                if Settings.MAP_ONE[y][x] == 0:
                    continue
                wall = Wall(Settings.WALLS[Settings.MAP_ONE[y][x]], self.screen)
                wall.rect.x = x*Settings.BOX_SIZE
                wall.rect.y = y*Settings.BOX_SIZE
                if Settings.MAP_ONE[y][x] == Settings.RED_WALL:
                    wall.type = Settings.RED_WALL
                elif Settings.MAP_ONE[y][x] == Settings.IRON_WALL:
                    wall.type = Settings.IRON_WALL
                elif Settings.MAP_ONE[y][x] == Settings.WEED_WALL:
                    wall.type = Settings.WEED_WALL
                elif Settings.MAP_ONE[y][x] == Settings.BOSS_WALL:
                    wall.type = Settings.BOSS_WALL
                    wall.life = 1
                self.walls.add(wall)

    def __check_keydown(self, event):
        """检查按下按钮的事件"""
        if event.key == pygame.K_LEFT:
            # 按下左键
            self.hero.direction = Settings.LEFT
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
            self.hero.terminal+=50
        elif event.key == pygame.K_RIGHT:
            # 按下右键
            self.hero.direction = Settings.RIGHT
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
            self.hero.terminal+=50
        elif event.key == pygame.K_UP:
            # 按下上键
            self.hero.direction = Settings.UP
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
            self.hero.terminal+=50
        elif event.key == pygame.K_DOWN:
            # 按下下键
            self.hero.direction = Settings.DOWN
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
            self.hero.terminal+=50
        elif event.key == pygame.K_SPACE:
            # 坦克发子弹
            self.hero.shot()

    def __check_keyup(self, event):
        """检查松开按钮的事件"""
        if event.key == pygame.K_LEFT:
            # 松开左键
            self.hero.direction = Settings.LEFT
            self.hero.is_moving = False
        elif event.key == pygame.K_RIGHT:
            # 松开右键
            self.hero.direction = Settings.RIGHT
            self.hero.is_moving = False
        elif event.key == pygame.K_UP:
            # 松开上键
            self.hero.direction = Settings.UP
            self.hero.is_moving = False
        elif event.key == pygame.K_DOWN:
            # 松开下键
            self.hero.direction = Settings.DOWN
            self.hero.is_moving = False

    def __event_handler(self):
        self.UpdateState()
        if(self.hero.terminal <= 0) or (self.hero.is_hit_wall):
            # self.__Random_agent()
            self.__Expectimax_agent()
            ''''''
        
        for event in pygame.event.get():
            # 判断是否是退出游戏
            if event.type == pygame.QUIT:           
                TankWar.__game_over()
            elif len(self.enemies) == 0:
                Settings.SCORE += 500
                TankWar.__win()
            if(self.hero.terminal <= 0):
                self.hero.is_moving = False
                if event.type == pygame.KEYDOWN:
                    TankWar.__check_keydown(self, event)
                elif event.type == pygame.KEYUP:
                    TankWar.__check_keyup(self, event)
                ''''''
    
    # 我的agent
    def __Expectimax_agent(self):
        if(len(self.enemies) == 0):
            self.hero.direction = random.choice(self.state.getActions(-1))
            return

        self.depth = 1
        initDepth = 1
        agentIndex = -1
        # 每次加50是希望玩家整格移动，化简问题
        self.hero.terminal += 50
        self.hero.is_moving = True
        self.hero.direction = self.expectiMax(self.state, initDepth, agentIndex)
        
        self.hero.random_shot()

    # 选择value最大的动作（默认敌人不转向）
    def expectiMax0(self, state, depth, agentIndex):        
           
        if (depth > self.depth):
            return state.evaluate()
        actions = state.getActions(agentIndex)
        states = state.nextStates(agentIndex)
        scores = [self.expectiMax(state, depth + 1, agentIndex) for state in states]
        if(len(scores) == 0):
            bestScore = state.evaluate(agentIndex)
        else:
            bestScore = max(scores)
        if(depth == 1):
            bestIndex = [index for index in range(len(scores)) if scores[index] == bestScore]
            bestAction = actions[random.choice(bestIndex)]
            return bestAction
        else:
            return bestScore
        
    # 选择value最大的动作（考虑敌人的随机转向）
    def expectiMax(self, state, depth, agentIndex):   
        nextDepth = depth
        nextAgent = agentIndex + 1
        if(nextAgent == len(self.enemies)):
            nextAgent = -1
        if(nextAgent == -1):
            nextDepth += 1     
           
        if (nextDepth > self.depth):
            return state.evaluate()
        actions = state.getActions(agentIndex)
        states = state.nextStates(agentIndex)
        if(agentIndex == -1):
            scores = [self.expectiMax(state, nextDepth, nextAgent) for state in states]
            if(len(scores) == 0):
                bestScore = state.evaluate()
            else:
                bestScore = max(scores)
            if(depth == 1):
                bestIndex = [index for index in range(len(scores)) if scores[index] == bestScore]
                bestAction = actions[random.choice(bestIndex)]
                return bestAction
            else:
                return bestScore
        else:
            sum = 0
            scores = [self.expectiMax(state, nextDepth, nextAgent) for state in states]
            if len(scores) == 0:
                return state.evaluate()
            else:
                for score in scores:
                    sum += score
                if(nextAgent >= len(self.enemies)):
                    return sum/len(scores)
                else:
                    sum += 4 * len(scores) * self.expectiMax(state.nextState(state.enemies[nextAgent][1], nextAgent), nextDepth, nextAgent)
                    return sum/(len(scores) * 5)
            
        
    # 随机agnet
    def __Random_agent(self):
        # NewDirection = self.hero.RandomAction()
        
        self.hero.terminal += 50
        self.hero.direction = random.choice(self.state.getActions())
        self.hero.is_moving = True
        self.hero.random_shot()
    
    def __check_collide(self):
        # 保证坦克不移出屏幕
        self.hero.hit_wall()
        for enemy in self.enemies:
            enemy.hit_wall_turn()
        # 子弹击中墙
        for wall in self.walls:
            # 我方英雄子弹击中墙
            for bullet in self.hero.bullets:
                if pygame.sprite.collide_rect(wall, bullet):
                    if wall.type == Settings.RED_WALL:
                        wall.kill()
                        bullet.kill()
                    elif wall.type == Settings.BOSS_WALL:
                        Settings.SCORE -= 500
                        self.game_still = False
                    elif wall.type == Settings.IRON_WALL:
                        bullet.kill()
            # 敌方英雄子弹击中墙
            for enemy in self.enemies:
                for bullet in enemy.bullets:
                    if pygame.sprite.collide_rect(wall, bullet):
                        if wall.type == Settings.RED_WALL:
                            wall.kill()
                            bullet.kill()
                        elif wall.type == Settings.BOSS_WALL:
                            Settings.SCORE -= 500
                            self.game_still = False
                        elif wall.type == Settings.IRON_WALL:
                            bullet.kill()
                            
            # 我方坦克撞墙
            if pygame.sprite.collide_rect(self.hero, wall):
                # 不可穿越墙
                if wall.type == Settings.RED_WALL or wall.type == Settings.IRON_WALL or wall.type == Settings.BOSS_WALL:
                    self.hero.is_hit_wall = True
                    # 移出墙内
                    self.hero.move_out_wall(wall)

            # 敌方坦克撞墙
            for enemy in self.enemies:
                if pygame.sprite.collide_rect(wall, enemy):
                    if wall.type == Settings.RED_WALL or wall.type == Settings.IRON_WALL or wall.type == Settings.BOSS_WALL:
                        enemy.move_out_wall(wall)
                        enemy.random_turn()

        # 子弹击中、敌方坦克碰撞、敌我坦克碰撞
        pygame.sprite.groupcollide(self.hero.bullets, self.enemies, True, True)
        # 敌方子弹击中我方
        for enemy in self.enemies:
            for bullet in enemy.bullets:
                if pygame.sprite.collide_rect(bullet, self.hero):
                    bullet.kill()
                    self.hero.kill()
                    return

    def __update_sprites(self):
        if self.hero.is_moving:
            self.hero.update()
        self.walls.update()
        self.hero.bullets.update()
        self.enemies.update()
        for enemy in self.enemies:
            enemy.bullets.update()
            enemy.bullets.draw(self.screen)
        self.enemies.draw(self.screen)
        self.hero.bullets.draw(self.screen)
        self.screen.blit(self.hero.image, self.hero.rect)
        self.walls.draw(self.screen)

    def run_game(self):
        self.__init_game()
        self.__create_sprite()
        enemy_num = Settings.ENEMY_COUNT
        frame = 0
        while True and self.hero.is_alive and self.game_still:
            if frame >= 30:
                frame = 0
                Settings.SCORE -= 1
            else:
                frame += 1
            self.screen.fill(Settings.SCREEN_COLOR)
            # 1、设置刷新帧率
            self.clock.tick(Settings.FPS)
            # 2、事件监听
            self.__event_handler()
            # 3、碰撞监测
            self.__check_collide()
            if(len(self.enemies) < enemy_num):
                Settings.SCORE += (enemy_num - len(self.enemies))*100
                enemy_num = len(self.enemies)
            # 4、更新/绘制精灵/经理组
            self.__update_sprites()
            # 5、更新显示
            pygame.display.update()
        self.__game_over()

    @staticmethod
    def __game_over():
        print("You are lost. Your Score is :", Settings.SCORE)
        pygame.quit()
        exit()
        
    def __win():
        print("Congraduations! Your Score is :", Settings.SCORE)
        pygame.quit()
        exit()