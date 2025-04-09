import json
from enum import IntEnum
from random import randrange
import functools

from core import ProblemElement, Action, ActionInput, ActionUtilityLevel


class Mapper:

    def __init__(self, json_path=None):
        self.data = None
        if isinstance(json_path, str):
            with open(json_path) as file:
                self.data = json.load(file)
        else:
            print(f'===> invalid json path')

    def get_expectations(self):
        if self.data is not None:
            return self.data.get('expectations')

        return None

    def get_reasoning_failures(self):
        if self.data is not None:
            return self.data.get('reasoning_failures')

        return None

    def recommendations(self):
        if self.data is not None:
            return self.data.get('recommendations')

        return None

    def detect_anomaly(self, action, attribute, pattern):
        if self.data is not None:
            expectations = self.get_expectations()
            results = list(filter(lambda exp: exp.get(
                'input').get('attribute') == attribute, expectations))
            if len(results) > 0:
                expectation = results[0]
                value = getattr(action, attribute)
                if pattern is not None and isinstance(value, list):
                    value = value[-len(pattern):]
                constraint = expectation.get('input').get('constraint')
                expected_value = expectation.get('input').get('expected_value')

                if self.compare(value, constraint, expected_value):
                    return (expectation, True)
            else:
                print(f'===> attribute {attribute} not found within json file')

        return (None, False)

    def compare(self, value, constraint, expected_value):
        if constraint == '=':
            if isinstance(value, list) and isinstance(expected_value, list):
                return self.compare_list(value, expected_value)
            else:
                return value == expected_value
        if constraint == '!=':
            if isinstance(value, list) and isinstance(expected_value, list):
                return self.compare_list(value, expected_value)
            else:
                return value != expected_value

        return False

    def compare_list(self, value, expected_value):
        if functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, value, expected_value), True):
            return True

        return False

    def get_recommendations(self, reasoning_failure):
        results = reasoning_failure.get('recommendations')
        for recommendation_id in results:
            r = self.recommendations()
            return list(
                filter(lambda item: item.get('id') == recommendation_id, r))


class IntershipTask:

    def __init__(self, name):
        self.id = 0
        self.name = name
        self.skills = []
        self.team = None
        self.team_size = 0

    def __str__(self):
        return f'id={self.id} and name={self.name}'


class Intership:

    def __init__(self, name):
        self.id = 0
        self.name = name
        self.tasks = []

    def __str__(self):
        return f'id={self.id} and name={self.name}'


class Skill:

    def __init__(self, name):
        self.id = 0
        self.name = name

    def __str__(self):
        return f'id={self.id} and name={self.name}'


class SkillLevel(IntEnum):
    HIGH = 5
    MEDIUM = 3
    LOW = 1


class SkillMastery:

    def __init__(self):
        self.mastery = SkillLevel.LOW
        self.skill = None

    def __str__(self):
        return f'{self.mastery}'


class Student:

    def __init__(self, id, name, skills_mastery):
        self.id = id
        self.name = name
        self.skills_mastery = skills_mastery
        self.quality = 0

    def update_quality(self):
        total = 0
        average = 0
        for skill_mastery in self.skills_mastery:
            total += skill_mastery.mastery
        average = total / len(self.skills_mastery)
        self.quality = average

    def __str__(self):
        return f'[Student] -> name={self.name} quality={self.quality}'


class Team(ProblemElement):

    def __init__(self):
        self.members = []

    def __str__(self):
        return f'[Team] [{str(self.members[0])}, {str(self.members[1])}, {str(self.members[2])}]'


class PickUpStudentWithPerfectScore(Action):

    def __init__(self, name):
        super().__init__(name)

    def run(self):
        student_universe = self.inputs[0]
        students = student_universe.students
        while self.output is None:
            r = list(filter(lambda s: self.constraint(s), students))
            student = r[0] if len(r) > 0 else None
            self.output = student
            self.success = student is not None
            self.action_completed = self.success
            self.profiles.append(self.success)

    def stop(self):
        pass

    def __del__(self):
        self.output = -1

    def get_observation_success_pattern(self):
        return [False, False]


class PickUpStudentByConstraint(Action):

    def __init__(self, name):
        super().__init__(name)

    def run(self):
        student_universe = self.inputs[0]
        students = student_universe.students
        r = list(filter(lambda s: self.constraint(s), students))
        student = r[0] if len(r) > 0 else None
        self.output = student
        self.success = student is not None
        self.profiles.append(self.success)

    def get_observation_success_pattern(self):
        return [False, False]


class PickUpStudentRandomly(Action):

    def __init__(self, name):
        super().__init__(name)

    def run(self):
        student_universe = self.inputs[0]
        students = student_universe.students
        student = students[randrange(len(students)) - 1]
        self.output = student
        self.success = self.constraint(student)
        self.profiles.append(self.success)

    def get_observation_success_pattern(self):
        return [False, False]


class StudentUniverse(ActionInput):

    def __init__(self):
        self.students = []


def get_skills():
    skill_1 = Skill(name='adobe-suite-skill')
    skill_2 = Skill(name='microsoft-word-skill')
    skill_3 = Skill(name='css3-skill')
    skill_4 = Skill(name='python3-skill')
    skill_5 = Skill(name='javascript-skill')
    return [skill_1, skill_2, skill_3, skill_4, skill_5]


def generate_skill_mastery(skills):
    levels = [SkillLevel.LOW, SkillLevel.MEDIUM, SkillLevel.HIGH]
    result = []
    for skill in skills:
        mastery = SkillMastery()
        mastery.mastery = levels[randrange(len(levels)) - 1]
        mastery.skill = skill
        result.append(mastery)

    return result


def generate_students(total, skills):
    students = []
    for item in range(total):
        default_masteries = generate_skill_mastery(skills)
        student = Student(
            id=item + 1,
            name=f'student-{item + 1}',
            skills_mastery=default_masteries
        )
        student.update_quality()
        students.append(student)

    return students


def choose_high_constraint(student):
    o = list(filter(lambda skill: skill.mastery ==
                    SkillLevel.HIGH, student.skills_mastery))
    return len(o) > 2


def choose_two_high_and_one_medium_constraint(student):
    m = list(filter(lambda skill: skill.mastery ==
                    SkillLevel.MEDIUM, student.skills_mastery))
    h = list(filter(lambda skill: skill.mastery ==
                    SkillLevel.HIGH, student.skills_mastery))
    return len(m) > 1 and len(h) > 1


def choose_medium_constraint(student):
    o = list(filter(lambda skill: skill.mastery ==
                    SkillLevel.MEDIUM, student.skills_mastery))
    return len(o) > 2


def create_sample_actions(universe):
    s_input = StudentUniverse()
    s_input.students = universe

    action1 = PickUpStudentRandomly(name='action-1')
    action1.utility = ActionUtilityLevel.FIVE
    action1.add_inputs(s_input)
    action1.add_constraint(choose_high_constraint)

    action2 = PickUpStudentRandomly(name='action-2')
    action2.utility = ActionUtilityLevel.THREE
    action2.add_inputs(s_input)
    action2.add_constraint(choose_two_high_and_one_medium_constraint)

    action3 = PickUpStudentRandomly(name='action-3')
    action3.utility = ActionUtilityLevel.TWO
    action3.add_inputs(s_input)
    action3.add_constraint(choose_medium_constraint)

    return [action1, action2, action3]
