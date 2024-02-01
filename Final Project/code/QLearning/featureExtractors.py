import settings
import util



def getFeatures(state, action):
    feature = util.Counter()
    BossWall = (9*50, 12*50)
    hero = state.hero
    enemy = state.enemies
    bullet = state.bullets
    distance1 = [util.manhattanDistance((hero.rect.centerx, hero.rect.centery), (oneEnemy.rect.centerx, oneEnemy.rect.centery)) for oneEnemy in enemy]
    distance2 = [util.manhattanDistance((hero.rect.centerx, hero.rect.centery), (onebullet.rect.centerx, onebullet.rect.centery)) for onebullet in bullet]
    distance = distance1 + distance2
    feature['distance_hero_enemy'] = min(distance)
    
    distance1 = [util.manhattanDistance(BossWall, (oneEnemy.rect.centerx, oneEnemy.rect.centery)) for oneEnemy in enemy]
    distance2 = [util.manhattanDistance(BossWall, (onebullet.rect.centerx, onebullet.rect.centery)) for onebullet in bullet]
    distance = distance1 + distance2
    
    
    feature['distance_boss_enemy'] = min(distance)
    
    return feature

