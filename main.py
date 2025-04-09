import time
from threading import Thread
from random import randrange

from core import AnytimeAlgorithm, Solution, Problem, CarinaModel, Memory,\
    PerformanceProfile, FTAP, StopReasoningProblem, \
    AllocateDeliverationTimeProblem, AnytimeAlgorithmPerformanceLevel, \
    EvaluationEffortProblem, Task, ReasoningFailureProblem
from miscellaneous import IntershipTask, Intership, Team, Skill, SkillMastery,\
    SkillLevel, Student, create_sample_actions, get_skills, generate_students, \
    create_sample_actions, Mapper
from predictor import rfc


# globals
skills = get_skills()
members_universe = generate_students(100, skills)
mapper = Mapper('./failures.json')
Memory()


class AllocationTeamProblem(FTAP):

    def __init__(self):
        self.reasoning_unit_of_time = 0.125

        # skills
        self.skills = skills

        # tasks
        task_1 = IntershipTask(name='graphic design')
        task_1.team_size = 3
        task_1.skills = [self.skills[0], self.skills[1], self.skills[2]]

        task_2 = IntershipTask(name='software developer')
        task_2.team_size = 3
        task_2.skills = [self.skills[3], self.skills[4]]
        self.tasks = [task_1, task_2]

        # intership
        intership = Intership(name='Sengi SAS')
        intership.tasks = [task_1, task_2]
        self.interships.append(intership)

        self.members_universe = members_universe


class AnytimeFTAP(AnytimeAlgorithm):
    INITIAL_COST_TIME = 0.03
    COST_TIME_FACTOR = 0.25

    def __init__(self, name):
        super().__init__(name)

        self.problem = AllocationTeamProblem()
        self.performance = PerformanceProfile()
        self.performance.algorithm = self
        self.performance.problem = self.problem

        Memory.instance.values.append(self.performance)

    def process(self):
        # create default solution
        self.solution = self.create_default_solution()
        times = len(self.performance.qualities)
        self.performance.qualities.append([self.solution.quality])

        # loop
        i = 0
        while i < self.time_constraint and self.stop_reasoning is False:
            print(f'===> [{self.name} Loop] -> {i}')
            # for now we just pick up the first one
            # TODO: create function to pick any solution
            team = self.solution.solutions[0]
            # print(f'{str(team)}')

            # pick up the worst student
            the_worst_index = self.pick_up_the_worst(team)
            the_worst = team.members[the_worst_index]

            # pick up a random student from the universe
            random_student = self.pick_up_random_student_from_universe()

            # compare the worst and a random student
            # the random is better that the worst
            if self.compare_students(random_student, the_worst):
                # remove the worst from the team
                # add the random student into the team
                team.members[the_worst_index] = random_student

                # update performance profile
                self.solution.quality = self.calc_team_quality(team)
                self.performance.qualities[times].append(self.solution.quality)
            else:
                # we add the latest quality created again
                latest_quality_saved = self.performance.qualities[times][-1]
                self.performance.qualities[times].append(latest_quality_saved)

            # increment
            i += 1
            self.reasoning_time = i
            time.sleep(self.problem.reasoning_unit_of_time)

    def run(self):
        # print(f'===> [RUN] -> {self.name}')
        self.process()

    def calc_team_quality(self, team):
        team_total = 0
        team_average = 0
        for member in team.members:
            member.update_quality()
            team_total += member.quality
        team_average = team_total / len(team.members)

        return team_average

    def create_default_solution(self):
        solution = Solution()
        for intership in self.problem.interships:
            for task in intership.tasks:
                team = self.create_default_team(
                    task.team_size,
                    self.problem.members_universe
                )
                solution.solutions.append(team)
                solution.quality = self.calc_team_quality(team)
                task.team = team

        return solution

    def create_default_team(self, total, students):
        team = Team()
        for _ in range(total):
            # TODO: don't choose the same student twice
            team.members.append(
                students[randrange(len(students)) - 1]
            )
        return team

    def pick_up_the_worst(self, team):
        team_sorted = sorted(team.members, key=lambda student: student.quality)
        lower = team_sorted[0]
        index = next((i for i, item in enumerate(
            team.members) if item.id == lower.id), -1)
        return index

    def pick_up_random_student_from_universe(self):
        action = self.problem.get_current_action()
        if isinstance(action, Task):
            action.run()
            # test = self.problem.get_test_for_action(action.name)
            # test.update_quality_history()
            # self.problem.update_test(test)
            # print(f'===> current action={str(action)}')
            return action.output
        else:
            # TODO: catch failure and choose another Action randomly
            pass

        return None

    def compare_students(self, student_a, student_b):
        return student_a.quality > student_b.quality

    def time_dependent_utility(self, q, t):
        u = self.intrinsec_utility(q) - self.cost_of_time(t)
        print(f'===> UTILITY FUNCTION={u}')
        return u

    def intrinsec_utility(self, q):
        return q

    def cost_of_time(self, index):
        return index * self.INITIAL_COST_TIME + self.COST_TIME_FACTOR


def first_metareasoning_problem():
    available_time_constraint = 200

    # algorithm
    # algorithm = AnytimeFTAP(name='alg-1')
    # algorithm.time_constraint = available_time_constraint

    # metareasoning problem
    reasoning_problem = StopReasoningProblem(name='stopping-reasoning-problem')
    reasoning_problem.monitoring_unit_of_time = 0.25
    reasoning_problem.available_time_constraint = available_time_constraint

    # feed and start carina
    carina = CarinaModel()

    actions = create_sample_actions(members_universe)
    for _ in range(11):
        algorithm = AnytimeFTAP(name=f'alg-1')
        algorithm.time_constraint = available_time_constraint

        carina.setup_stopping_reasoning_problem(reasoning_problem, algorithm)
    carina.training(1, actions[0])


def second_metareasoning_problem(a, b, c):
    # requirements
    training_time_constraint = int(a)
    available_time_constraint = int(b)
    total_algorithm_in_parallel = int(c)

    # algorithms
    algorithm1 = AnytimeFTAP(name='alg-1')
    algorithm1.time_constraint = training_time_constraint
    algorithm2 = AnytimeFTAP(name='alg-2')
    algorithm2.time_constraint = training_time_constraint
    algorithm3 = AnytimeFTAP(name='alg-3')
    algorithm3.time_constraint = training_time_constraint

    # metareasoning problem
    stopping_reasoning_problem = StopReasoningProblem(
        name='stopping-reasoning-problem')
    stopping_reasoning_problem.monitoring_unit_of_time = 0.25
    stopping_reasoning_problem.available_time_constraint = available_time_constraint

    reasoning_problem = AllocateDeliverationTimeProblem(
        name='allocate-deliveration-time-problem')
    reasoning_problem.monitoring_unit_of_time = 0.25
    reasoning_problem.time_constraint = training_time_constraint
    reasoning_problem.available_time_constraint = available_time_constraint
    reasoning_problem.total_algorithm_in_parallel_constraint = total_algorithm_in_parallel
    reasoning_problem.algorithms = [
        algorithm1,
        algorithm2,
        algorithm3,
    ]

    evaluation_effort_problem = EvaluationEffortProblem(
        name='evaluation-effort-problem')
    evaluation_effort_problem.monitoring_unit_of_time = 0.25
    actions = create_sample_actions(members_universe)
    for action in actions:
        evaluation_effort_problem.add_action(action)

    reasoning_failure_problem = ReasoningFailureProblem(
        name='reasoning-failure-problem')
    reasoning_failure_problem.monitoring_unit_of_time = 0.25
    reasoning_failure_problem.mapper = mapper

    # feed and start carina
    carina = CarinaModel(set_stop_reasoning_predictor=rfc)
    carina.setup_allocate_deliveration_time_problem(reasoning_problem)
    carina.setup_stopping_reasoning_problem(stopping_reasoning_problem)
    carina.setup_evaluation_effort_problem(evaluation_effort_problem)
    carina.setup_reasoning_failure_problem(reasoning_failure_problem)
    carina.start()


if __name__ == '__main__':
    x = input(
        "Do you want to add a time restriction into the reasoning process? (y/n)")
    if x == 'y':
        a = input("Type training time constraint (number):")
        b = input("Type available time constraint (number): ")
        if int(b) > int(a):
            print(
                '"training time constraint" MUST BE GREATER than "available time constraint"')
        else:
            c = input("Type total algorithm in parallel (number):")
            if int(c) <= 3:
                second_metareasoning_problem(a, b, c)
    else:
        first_metareasoning_problem()
