import json
import csv
from random import randrange

from core import MentalState, Goal, ModelOfTheWorld, Planner, PlanningProblem, \
    Action, ModelOfTheSelf
from miscellaneous import mapping, mapping_2


class KillEnemyGame(PlanningProblem):

    def __init__(self):
        super().__init__('kill-enemy-game')


def test_game():
    ModelOfTheSelf()
    ModelOfTheWorld()

    # mental states
    s1 = MentalState('kill_enemy', False)
    s2 = MentalState('loaded_weapon', False)
    s3 = MentalState('taken_weapon', False)
    s4 = MentalState('shoot_weapon', False)
    s5 = MentalState('drawed_weapon', False)

    # actions
    a1 = Action(name='draw-weapon-action')
    a1.preconditions = [(s5, False)]
    a1.effects = [(s5, True)]

    a2 = Action(name='load-weapon-action')
    a2.preconditions = [(s3, True)]
    a2.effects = [(s2, True)]

    a3 = Action(name='take-weapon-action')
    a3.preconditions = [(s5, True)]
    a3.effects = [(s3, True)]

    a4 = Action(name='shoot-weapon-action')
    a4.preconditions = [(s2, True)]
    a4.effects = [(s1, True)]

    desired_state = MentalState('kill_enemy', True)

    game = KillEnemyGame()
    game.load_states([s1, s2, s3, s4, s5])
    game.desired_states = [desired_state]
    game.actions = [a1, a2, a3, a4]

    planner = Planner()
    planner.problem = game
    planner.start()
    planner.solve()
    planner.show_plan()

# ==================================================== temp


def build_dataset():
    data = []
    with open('./declarative-store.json') as file:
        data = json.load(file)

        print(f'===> len {len(data)}')
        positives = list(filter(lambda obj: obj.get('why').get(
            'state')[0].get('value') is True, data))
        print(f'===> len f {len(positives)}')

        negatives = list(filter(lambda obj: obj.get('why').get(
            'state')[0].get('value') is False, data))

        with open('./dataset.json') as outfile:
            dataset = json.load(outfile)

            # map positives
            mapped = list(map(lambda o: [
                o.get('when').get('state')[0].get('value'),
                o.get('when').get('state')[1].get('value'),
                o.get('when').get('state')[2].get('value'),
                o.get('when').get('state')[3].get('value'),
                o.get('when').get('state')[4].get('value'),
                o.get('when').get('state')[5].get('value'),
                o.get('when').get('state')[6].get('value'),
                o.get('when').get('state')[8].get('value'),
                o.get('action').get('type'),
            ], positives))
            dataset += mapped

            # map negatives
            randomly_selected = []
            # INFO: mapped is 100 length
            for _ in range(len(mapped)):
                r = randrange(len(negatives) - 1)
                randomly_selected.append(
                    negatives[r]
                )

            mapped = list(map(lambda o: [
                o.get('when').get('state')[0].get('value'),
                o.get('when').get('state')[1].get('value'),
                o.get('when').get('state')[2].get('value'),
                o.get('when').get('state')[3].get('value'),
                o.get('when').get('state')[4].get('value'),
                o.get('when').get('state')[5].get('value'),
                o.get('when').get('state')[6].get('value'),
                o.get('when').get('state')[8].get('value'),
                o.get('action').get('type'),
            ], randomly_selected))
            dataset += mapped

            print(f'===> total-dataset {len(dataset)}')

            # save dataset
            with open('./dataset.json', 'w') as writable_file:
                json.dump(dataset, writable_file, indent=2)


def json_to_csv():
    with open('./dataset.json') as outfile:
        dataset = json.load(outfile)

        with open('./dataset.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obj in dataset:
                writer.writerow([
                    obj[0],
                    obj[1],
                    obj[2],
                    obj[3],
                    obj[4],
                    mapping(obj[5]),
                    mapping_2(obj[6]),
                    obj[7],
                    obj[8]
                ])


# build_dataset()
# json_to_csv()
