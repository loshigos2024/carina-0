import time
from threading import Thread
from enum import IntEnum
from random import randrange
import json

import matplotlib.pyplot as plt
from conditional_knowledge_engine import ConditionalKnowledgeEngine


def mapping(value):
    if value == 'action-1':
        return 0
    if value == 'action-2':
        return 1
    if value == 'action-3':
        return 2


def mapping_2(value):
    if value == 'alg-1':
        return 0
    if value == 'alg-2':
        return 1
    if value == 'alg-3':
        return 2


class Problem:

    def __init__(self, name):
        self.name = name
        self.solution = None
        self.problem_elements = []


class PlanningProblem(Problem):

    def __init__(self, name):
        super().__init__(name)
        self.initial_states = []
        self.desired_states = []
        self.goales = []
        self.actions = []

    def load_states(self, states):
        self.initial_states = states
        ModelOfTheWorld.instance.mental_states = states


class MetareasoningProblem(Problem):

    def __init__(self, name):
        super().__init__(name)
        self.monitoring_unit_of_time = 0.0
        self.algorithm = None


class ReasoningProblem(Problem):

    def __init__(self, name):
        super().__init__(name)
        self.reasoning_unit_of_time = 0.0

    def get_current_action(self):
        return ModelOfTheSelf.instance.current_action

    def get_test_for_action(self, task_name):
        all_test = ModelOfTheSelf.instance.utility_of_evaluation_test
        result = list(
            filter(lambda test: test.action.name == task_name, all_test))
        return result[0] if len(result) > 0 else None

    def update_test(self, updated_test):
        all_test = ModelOfTheSelf.instance.utility_of_evaluation_test

        result = list(
            filter(lambda test: test[1].action.name ==
                   updated_test.action.name, enumerate(all_test)
                   ))

        index = result[0][0]
        ModelOfTheSelf.instance.utility_of_evaluation_test[index] = updated_test


class FTAP(ReasoningProblem):
    interships = []
    members_universe = []
    skills = []
    tasks = []


class StopReasoningProblem(MetareasoningProblem):

    def __init__(self, name):
        super().__init__(name)
        self.available_time_constraint = 0

    def stop_reasoning(self):
        if isinstance(self.algorithm, AnytimeAlgorithm):
            self.algorithm.stop()


class AllocateDeliverationTimeProblem(MetareasoningProblem):

    def __init__(self, name):
        super().__init__(name)
        self.time_constraint = 0
        self.available_time_constraint = 0
        self.total_algorithm_in_parallel_constraint = 0
        self.algorithms = []

    def training(self, times):
        for algorithm in self.algorithms:
            for _ in range(times):
                algorithm.run()

    def calc_solution_quality(self):
        # legends = []
        for algorithm in self.algorithms:
            first_qualities_cycle = len(algorithm.performance.qualities[0])

            for row in range(first_qualities_cycle):
                # print(f'==> row={row}')
                total = 0
                average = 0
                row_values = []
                for qualities in algorithm.performance.qualities:
                    total += qualities[row]
                    row_values.append(qualities[row])
                average = total / len(algorithm.performance.qualities)
                # m = '+'.join(map(str, row_values))
                # print(f'===> values={m} | average={average}')
                algorithm.performance.h.append(average)

            # line, = plt.plot(
            #     algorithm.performance.h,
            #     label=algorithm.name,
            #     linewidth=1.0,
            #     linestyle='--'
            # )
            # legends.append(line)
            # print(f'==> {algorithm.name}={algorithm.performance.h}')

        # plt.ylabel('quality')
        # plt.xlabel('unit of time')
        # plt.legend(
        #     handles=legends,
        #     loc='upper center',
        #     title='Algorithms',
        #     bbox_to_anchor=(0.5, 1.15),
        #     ncol=3,
        #     fancybox=False,
        #     shadow=False,
        # )
        # # plt.title('CARINA')
        # plt.grid(True)
        # # plt.show()

    def resolve_algorithm(self):
        if self.total_algorithm_in_parallel_constraint < len(self.algorithms):
            row = self.available_time_constraint
            for algorithm in self.algorithms:
                average = algorithm.performance.h[row]
                algorithm.hq = average
            candidates = sorted(
                self.algorithms, key=lambda algorithm: -algorithm.hq)

            return candidates[0:self.total_algorithm_in_parallel_constraint]
        else:
            print('===> parallel algorithms out of bound')

        return [self.algorithms[0]]


class EvaluationEffortProblem(MetareasoningProblem):

    def __init__(self, name):
        super().__init__(name)
        self.actions = []
        self.max_test_to_do = 2

    def add_action(self, new_action):
        self.actions.append(new_action)


class ReasoningFailureProblem(MetareasoningProblem):

    def __init__(self, name):
        super().__init__(name)
        self.mapper = None


class ReasoningFailure:

    def __init__(self, data, expectation):
        self.data = data
        self.expectation = expectation

    def find_reasoning_failure(self):
        if isinstance(self.expectation, dict):
            id = self.expectation.get('id')
            reasoning_failures = self.data.get('reasoning_failures')
            rf_found = list(
                filter(lambda r: id in r.get('causes'), reasoning_failures))
            if len(rf_found) > 0:
                return rf_found[0]

        return None


class Task:

    def __init__(self, name):
        self.name = name
        self.preconditions = []
        self.effects = []
        self.constraint = None
        self.inputs = []
        self.output = None
        self.success = False
        self.profiles = []
        self.action_completed = False

    def add_inputs(self, new_input):
        self.inputs.append(new_input)

    def add_constraint(self, constraint):
        self.constraint = constraint
        # TODO: validate if callable
        # if callable(self.constraint):
        #     self.constraint = constraint
        # else:
        #     print(f'===> error: {constraint} is not a callable')

    def run(self):
        pass

    def get_observation_success_pattern(self):
        pass

    def __str__(self):
        return self.name


class ActionUtilityLevel(IntEnum):
    FIVE = 5
    FOUR = 4
    THREE = 3
    TWO = 2
    ONE = 1
    NONE = 0


class Action(Task):

    def __init__(self, name):
        super().__init__(name)
        self.utility = ActionUtilityLevel.NONE

    def run(self):
        pass

    def stop(self):
        pass


class ActionInput:
    pass


class EvaluationProfile:

    def __init__(self):
        pass


class EvaluationTest:

    def __init__(self):
        self.action = None
        self.quality_history = []

    def training(self, total_training):
        for _ in range(total_training):
            self.action.run()
            self.update_quality_history()

    def test(self):
        if isinstance(self.action, Task):
            return self.expected_utility()
        else:
            print('===> not action configured')

        return 0

    def expected_utility(self):
        # P(t1|a) a = self.calc_utility(1)
        # P(t1|-a) a2 = self.calc_utility(0)
        # P(a) c = self.prior_probability(True)
        # P(-a) c2 = self.prior_probability(False)
        # P(t1) w = P(t1|a) * P(a) + P(t1|-a) * P(-a)
        # P(a|t1) x = (P(t1|a) * P(a)) / P(t1)
        # EXP_UTL = self.action.utility * P(a|t1)

        a = self.calc_utility(1)
        a2 = self.calc_utility(0)
        c = self.prior_probability(True)
        c2 = self.prior_probability(False)

        w = a * c + a2 * c2
        x = (a * c) / w

        return int(self.action.utility) * x

    def prior_probability(self, criteria):
        result = list(filter(lambda value: value ==
                             criteria, self.action.profiles))
        return len(result) / len(self.action.profiles)

    def calc_utility(self, criteria):
        result = list(filter(lambda value: value ==
                             criteria, self.quality_history))
        return len(result) / len(self.quality_history)

    def update_quality_history(self):
        self.quality_history.append(1 if self.action.success else 0)

    def __str__(self):
        return f'test of {self.action.name}'


class Algorithm:

    def __init__(self, name):
        self.name = name
        self.solution = None
        self.is_running = False  # TODO: remove it?

    def run(self):
        pass

    def __str__(self):
        return self.name


class AnytimeAlgorithmPerformanceLevel(IntEnum):
    HIGH = 5
    MEDIUM = 3
    LOW = 1


class AnytimeAlgorithm(Algorithm):

    def __init__(self, name):
        super().__init__(name)
        self.stop_reasoning = False
        self.reasoning_time = 0
        self.time_constraint = 0
        self.performance_level = AnytimeAlgorithmPerformanceLevel.LOW
        self.hq = 0

    def stop(self):
        self.stop_reasoning = True
        print('===> stopped')


class PerformanceProfile:

    def __init__(self):
        self.problem = None
        self.algorithm = None
        self.qualities = []
        self.h = []

    def __str__(self):
        return f'{self.qualities}'


class Memory:
    class __Memory:
        def __init__(self):
            self.values = []
    instance = None

    def __init__(self):
        if not Memory.instance:
            Memory.instance = Memory.__Memory()
        else:
            Memory.instance.values = []

    def __getattr__(self, name):
        return getattr(self.instance, name)


class WorkingMemory:
    pass


class Planner:

    def __init__(self):
        self.problem = None
        self.plan = []

    def start(self):
        if isinstance(self.problem, PlanningProblem):
            for state in self.problem.desired_states:
                goal = Goal(state)
                ModelOfTheSelf.instance.goals.append(goal)

    def solve(self):
        while len(ModelOfTheSelf.instance.goals) > 0:
            goal = ModelOfTheSelf.instance.goals[-1]
            action = self.resolve(goal.desired_state)
            if action is not None:
                self.plan.append(action)
                if self.has_preconditions(action):
                    if self.resolve_preconditions(action) is False:
                        break
            else:
                break
            time.sleep(0.5)

    def has_preconditions(self, action):
        return len(action.preconditions) > 0

    def resolve_preconditions(self, action):
        for pre in action.preconditions:
            (a, b) = pre
            mental_states = ModelOfTheWorld.instance.mental_states
            found = list(filter(lambda s: s.name == a.name, mental_states))
            if len(found) > 0:
                if b != found[0].value:
                    a.value = b
                    new_goal = Goal(a)
                    ModelOfTheSelf.instance.goals.append(new_goal)
                    return True

        return False

    def resolve(self, state):
        for action in self.problem.actions:
            for effect in action.effects:
                (a, b) = effect
                if a.name == state.name and b == state.value:
                    return action

        return None

    def show_plan(self):
        self.plan.reverse()
        for p in self.plan:
            print(f'===> plan {p}')

    def watch(self):
        while True:
            goal = ModelOfTheSelf.instance.current_goal
            if goal is not None:
                pass
                # 1. recorrer el array de acciones
                # if action.effect == goal:
                # agregar la accion al plan
                # buscar la pre-condicion de la accion (es un estado)
                # action_shoot_weapon.effect (es un estado) [el estado es kill_enemy = True]
                # action_shoot_weapon.pre_condition (es un estado) [loaded_weapon = True]
                # se no se cumple la pre-condition, se crea un Goal nuevo y se agrega a a lista de goles y pasa a ser el current_goal
            # invertir la lista de goles


class MentalState:

    def __init__(self, name, value):
        self.name = name
        self.value = value


class Goal:

    def __init__(self, desired_state):
        self.desired_state = desired_state


class ModelOfTheWorld:
    class __ModelOfTheWorld:
        def __init__(self):
            self.plans = []
            self.mental_states = []
    instance = None

    def __init__(self):
        if not ModelOfTheWorld.instance:
            ModelOfTheWorld.instance = ModelOfTheWorld.__ModelOfTheWorld()

    def __getattr__(self, name):
        return getattr(self.instance, name)


class ModelOfTheSelf:
    class __ModelOfTheSelf:
        def __init__(self):
            self.reasoning_is_running = False
            self.algorithms_to_run = []
            self.current_algorithm = None
            self.current_profile = None
            self.is_resoning_completed = False
            self.current_action = None
            self.utility_of_evaluation_test = []
            self.goals = []  # motivational system
            self.current_goal = None
            # we have to move these
            self.cost_of_the_time = None
            self.reasoning_time = None
            self.intrinsec_utility = None
            self.stop_reasoning = None
            self.last_two_utility_function_values = []
            self.stop_reasoning_predictor = None

        def update_last_two_utility_function_values(self, new_value):
            if len(self.last_two_utility_function_values) < 2:
                self.last_two_utility_function_values.append(new_value)
            else:
                self.last_two_utility_function_values[0] = self.last_two_utility_function_values[-1]
                self.last_two_utility_function_values[1] = new_value

        def set_stop_reasoning_predictor(self, stop_reasoning_predictor):
            self.stop_reasoning_predictor = stop_reasoning_predictor

        def predict_stop_reasoning(self, input_data):
            print(f'=====> input_data: {input_data}')
            result = self.stop_reasoning_predictor.predict(input_data)
            print(f'=====> prediction outcome: {result[0]}')
            return result[0] == 'stop_reasoning'

    instance = None

    def __init__(self):
        if not ModelOfTheSelf.instance:
            ModelOfTheSelf.instance = ModelOfTheSelf.__ModelOfTheSelf()

    def __getattr__(self, name):
        return getattr(self.instance, name)


class ProblemElement:
    pass


class Solution:

    def __init__(self):
        self.quality = 0
        self.solutions = []


class ObjectLevel:

    def __init__(self):
        self.reasoning_problem = None

    def reasoning(self):
        print(f'===> [ObjectLevel REASONING STARTING]')
        for algorithm in ModelOfTheSelf.instance.algorithms_to_run:
            ModelOfTheSelf.instance.reasoning_is_running = True
            if isinstance(algorithm, Algorithm):
                ModelOfTheSelf.instance.current_algorithm = algorithm
                ModelOfTheSelf.instance.current_profile = algorithm.performance
                algorithm.time_constraint = self.reasoning_problem.available_time_constraint
                algorithm.reasoning_time = 0
                algorithm.run()
                ModelOfTheSelf.instance.reasoning_is_running = False
                ModelOfTheSelf.instance.current_algorithm = None
                ModelOfTheSelf.instance.current_profile = None

        ModelOfTheSelf.instance.is_resoning_completed = True


class MetaLevel:

    def __init__(self):
        self.stopping_reasoning_problem = None
        self.allocate_deliveration_time_problem = None
        self.evaluation_effort_problem = None
        self.reasoning_failure_problem = None
        self.data_to_plot = []
        self.total_time = []

    def meta_reasoning(self):
        if isinstance(self.allocate_deliveration_time_problem, AllocateDeliverationTimeProblem):
            problem = self.allocate_deliveration_time_problem
            self.process_allocate_deliveration_time_problem(problem)

        if isinstance(self.stopping_reasoning_problem, StopReasoningProblem):
            problem = self.stopping_reasoning_problem
            thread = Thread(
                target=self.process_stopping_metareasoning_problem,
                args=(problem,)
            )
            thread.start()

        if isinstance(self.evaluation_effort_problem, EvaluationEffortProblem):
            problem = self.evaluation_effort_problem
            thread = Thread(
                target=self.process_evaluation_effort_problem,
                args=(problem,)
            )
            # thread.start()

        if isinstance(self.reasoning_failure_problem, ReasoningFailureProblem):
            problem = self.reasoning_failure_problem
            # thread = Thread(
            #     target=self.process_reasoning_failure_problem,
            #     args=(problem,)
            # )
            # thread.start()

    def process_allocate_deliveration_time_problem(self, problem):
        # TODO: create predictor to pick up an alg
        problem.training(3)  # TODO: fix this fixed number
        problem.calc_solution_quality()
        ModelOfTheSelf.instance.algorithms_to_run = problem.resolve_algorithm()
        for i in ModelOfTheSelf.instance.algorithms_to_run:
            print(f'===> [SELECTED ALGORITHM] {str(i)}')

    def process_stopping_metareasoning_problem(self, problem):
        # TODO: if the reasoning is running
        print(f'===> process_stopping_metareasoning_problem While')
        while True:
            # l = len(Memory.instance.values)
            # print(f'===> L => {l}')
            # if l > 0:
            # TODO: look out with the [0]
            profile = ModelOfTheSelf.instance.current_profile
            algorithm = ModelOfTheSelf.instance.current_algorithm
            if isinstance(algorithm, AnytimeAlgorithm):
                if len(profile.qualities) > 0:
                    if len(profile.qualities[-1]) > 0:
                        q = profile.qualities[-1][-1]
                        t = algorithm.reasoning_time
                        print(f'===> q={q} t={t}')

                        ModelOfTheSelf.instance.cost_of_the_time = algorithm.cost_of_time(
                            t)
                        ModelOfTheSelf.instance.reasoning_time = algorithm.reasoning_time
                        ModelOfTheSelf.instance.intrinsec_utility = algorithm.intrinsec_utility(
                            q)
                        ModelOfTheSelf.instance.stop_reasoning = algorithm.stop_reasoning
                        action_name = ModelOfTheSelf.instance.current_action.name

                        a = [
                            {'name': 'q', 'value': q},
                            {'name': 't', 'value': t},
                            {'name': 'cost_of_the_time',
                                'value': ModelOfTheSelf.instance.cost_of_the_time},
                            {'name': 'intrinsec_utility',
                                'value': ModelOfTheSelf.instance.intrinsec_utility},
                            {'name': 'reasoning_time',
                                'value': ModelOfTheSelf.instance.reasoning_time},
                            {'name': 'action_name', 'value': action_name},
                            {'name': 'algorithm_name',
                                'value': algorithm.name},
                        ]
                        b = []
                        c = 'procedural'
                        d = ''

                        xu = algorithm.time_dependent_utility(q, t)
                        self.data_to_plot.append(xu)
                        self.total_time.append(t)
                        ModelOfTheSelf.instance.update_last_two_utility_function_values(
                            xu)

                        current_state_of_reasoning_process = [[
                            q,
                            t,
                            ModelOfTheSelf.instance.cost_of_the_time,
                            ModelOfTheSelf.instance.intrinsec_utility,
                            ModelOfTheSelf.instance.reasoning_time,
                            mapping(action_name),
                            mapping_2(algorithm.name),
                            xu
                        ]]
                        stop_reasoning = ModelOfTheSelf.instance.predict_stop_reasoning(
                            current_state_of_reasoning_process)
                        if stop_reasoning:
                            print('======> STOP')
                            last_positive_value = ModelOfTheSelf.instance.last_two_utility_function_values[
                                0]
                            b = [
                                {'name': 'stop_reasoning', 'value': True},
                            ]
                            d = 'stop_reasoning'
                            algorithm.stop()
                            a.append(
                                {'name': 'stop_reasoning', 'value': True})
                            a.append(
                                {'name': 'utility_function', 'value': last_positive_value})
                        else:
                            b = [
                                {'name': 'stop_reasoning', 'value': False},
                            ]
                            d = 'keep_reasoning'
                            a.append(
                                {'name': 'stop_reasoning', 'value': False})
                            a.append(
                                {'name': 'utility_function', 'value': xu})
                else:
                    print('===> profile qualities is empty')

                time.sleep(problem.monitoring_unit_of_time)
            else:
                print('===> Nothing running yet')

            if ModelOfTheSelf.instance.is_resoning_completed:
                print('===> Break while')
                break

    def create_plot(self, y):
        legends = []
        line, = plt.plot(
            self.total_time,
            y,
            label='expected utility',
            linewidth=1.0,
            linestyle='--'
        )
        legends.append(line)

        plt.ylabel('expected utility')
        plt.xlabel('unit of time')
        plt.legend(
            handles=legends,
            loc='upper center',
            title='Expected utility',
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fancybox=False,
            shadow=False,
        )
        # plt.title('CARINA')
        plt.grid(True)
        plt.show()

    def process_evaluation_effort_problem(self, problem):
        while True:
            # TODO:
            u = ModelOfTheSelf.instance.utility_of_evaluation_test
            print(
                f'===> t1={u[0].expected_utility()} t2={u[1].expected_utility()} t3={u[2].expected_utility()}')
            time.sleep(problem.monitoring_unit_of_time)

            if ModelOfTheSelf.instance.is_resoning_completed:
                print('===> Break while')
                break

    def process_reasoning_failure_problem(self, problem):
        while True:
            # TODO: leer current action y filtrar el MetaReasoningPoint
            action = ModelOfTheSelf.instance.current_action
            pattern = action.get_observation_success_pattern()

            # TODO: leer la observación

            # TODO: crear expectativas

            # TODO: verificar si hay violacion de expectativa(obervación)
            (expectation, expectation_violation) = problem.mapper.detect_anomaly(
                action, 'profiles', pattern)
            if expectation_violation:
                print('===> XXXXXXXXXXXXXXXXXXXXXXXXXX ===> 😎')
                # TODO: si hay falla de razonamiento, se crea un ReasoningFailure
                rf = ReasoningFailure(problem.mapper.data, expectation)
                failure = rf.find_reasoning_failure()
                recommendations = problem.mapper.get_recommendations(failure)
                # TODO: convert recommendation into Goal

            # TODO: identificar el tipo de ReasoningFailure

            # TODO: encontrar la causa de la falla(reasoingFailure)

            # TODO: dada la causa, se busca y selecciona una solución

            # TODO: salida -> goal | strategy | unknow

            # TODO: pasar el output al ModelOfTheSelf para ser leída
            # por el proceso de razonamiento

            time.sleep(problem.monitoring_unit_of_time)

            if ModelOfTheSelf.instance.is_resoning_completed:
                print('===> Break while')
                break

    def training_evaluation_effort(self):
        # training test
        u = []
        for action in self.evaluation_effort_problem.actions:
            et = EvaluationTest()
            et.action = action
            et.training(100)
            print(f'===> qh={et.quality_history}')
            print(f'===> eu={et.expected_utility()}')
            u.append(et)

        problem = self.evaluation_effort_problem
        self.update_current_action(u, problem)
        ModelOfTheSelf.instance.utility_of_evaluation_test = u

    def update_current_action(self, u, problem):
        # sort test
        # NOTE: this is a simulation of picking up unit of time constraint
        policy = []
        limit = problem.max_test_to_do
        while len(policy) < limit:
            test = u[randrange(len(u)) - 1]
            found = list(filter(lambda t: t.action.name ==
                                test.action.name, policy))
            if len(found) == 0:
                policy.append(test)

        test_sorted = sorted(policy, key=lambda test: -test.expected_utility())
        print(f's1={str(policy[0])} s2={str(policy[1])}')

        # pick up the best test
        selected_test = test_sorted[0]
        print(f'===> selected action={str(selected_test.action)}')

        # pass the action to the object level through the model of the self
        ModelOfTheSelf.instance.current_action = selected_test.action

    def run(self):
        self.meta_reasoning()


class CarinaModel:

    def __init__(self, set_stop_reasoning_predictor):
        # initialize memory
        Memory()

        # metareasoning problems to sove
        self.allocate_deliveration_time_problem = None
        self.stopping_reasoning_problem = None
        self.evaluation_effort_problem = None
        self.reasoning_failure_problem = None

        # knowledge engine
        # ConditionalKnowledgeEngine()

        # init model of the self
        ModelOfTheSelf()
        ModelOfTheSelf.instance.set_stop_reasoning_predictor(
            set_stop_reasoning_predictor)

        # set up object level
        self.object_level = ObjectLevel()

        # set up meta level
        self.metalevel = MetaLevel()

    def setup_allocate_deliveration_time_problem(self, reasoning_problem):
        self.allocate_deliveration_time_problem = reasoning_problem
        self.metalevel.allocate_deliveration_time_problem = reasoning_problem

    def setup_stopping_reasoning_problem(self, reasoning_problem, algorithm=None):
        self.stopping_reasoning_problem = reasoning_problem
        self.metalevel.stopping_reasoning_problem = reasoning_problem
        self.object_level.reasoning_problem = reasoning_problem

        if algorithm is not None:
            ModelOfTheSelf.instance.algorithms_to_run.append(algorithm)

    def setup_evaluation_effort_problem(self, reasoning_problem):
        self.evaluation_effort_problem = reasoning_problem
        self.metalevel.evaluation_effort_problem = reasoning_problem
        self.metalevel.training_evaluation_effort()

    def setup_reasoning_failure_problem(self, reasoning_problem):
        self.reasoning_failure_problem = reasoning_problem
        self.metalevel.reasoning_failure_problem = reasoning_problem

    def start(self):
        # ModelOfTheSelf.instance.current_action = action
        # ModelOfTheSelf.instance.current_algorithm = algorithm

        # run metareasoning
        self.metalevel.run()

        # run reasoning
        self.object_level.reasoning()

        self.metalevel.create_plot(self.metalevel.data_to_plot)
