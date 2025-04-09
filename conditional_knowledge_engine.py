import json


class Store:

    def __init__(self, json_path=None):
        self.json_path = json_path
        self.data = None

        if isinstance(self.json_path, str):
            with open(self.json_path) as file:
                self.data = json.load(file)
        else:
            print(f'===> invalid json path')

    def add_record(self, when_entry, why_entry, type, action):
        self.data.append({
            'when': {
                'state': when_entry
            },
            'why': {
                'state': why_entry
            },
            'type': type,
            'action': {
                'type': action
            }
        })

        self.update_json_file()

    def update_json_file(self):
        with open(self.json_path, 'w') as outfile:
            json.dump(self.data, outfile, indent=2)


class ConditionalKnowledgeEngine:
    class __ConditionalKnowledgeEngine:
        def __init__(self):
            self.store = Store(json_path='./declarative-store.json')

        def add_record(self, when_entry, why_entry, type, action):
            self.store.add_record(when_entry, why_entry, type, action)
    instance = None

    def __init__(self):
        if not ConditionalKnowledgeEngine.instance:
            ConditionalKnowledgeEngine.instance = ConditionalKnowledgeEngine.__ConditionalKnowledgeEngine()

    def __getattr__(self, name):
        return getattr(self.instance, name)
