import os
import json
import hashlib

from typing import Tuple


class NestedData:
    CLASS_SHORTHAND = None
    FIELDS = None
    NON_JSONABLE_FIELDS = ()

    def __init__(self,
                 *args,
                 **kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0

    def to_dict(self):
        dict_repr = dict()
        for field in self.FIELDS:
            if hasattr(self, field):
                field_val = getattr(self, field)
                if issubclass(type(field_val), Config) or issubclass(type(field_val), Schedule):
                    dict_repr[field] = field_val.to_dict()
                elif field in self.NON_JSONABLE_FIELDS or isinstance(field_val, complex):
                    dict_repr[field] = f'{field_val}'
                else:
                    dict_repr[field] = field_val
        return dict_repr

    def to_flat_dict(self):
        flat_dict_repr = dict()
        for field in self.FIELDS:
            if hasattr(self, field):
                field_val = getattr(self, field)
                if issubclass(type(field_val), Config):
                    flat_dict_repr.update(field_val.to_flat_dict())
                else:
                    flat_dict_repr[field] = field_val

        return flat_dict_repr

    def to_json_dict(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_json(self, filename: str = None):
        with open(filename, 'w') as f:
            json.dump(self.to_json_dict(), f)

    def to_path_suffix(self):
        path_suffix = []
        for field in self.FIELDS:
            if hasattr(self, field):
                field_val = getattr(self, field)
                if isinstance(field_val, Config):
                    path_suffix.append(field_val.to_path_suffix())
                else:
                    path_suffix.append(f'{field}={field_val}')

        return os.path.join(*path_suffix)

    def __hash__(self):
        return hash(self.to_json_dict())

    def to_sha256_str(self):
        hash_factory = hashlib.sha256()
        hash_factory.update(bytes(self.__str__(), 'ascii'))

        return hash_factory.hexdigest()

    def __repr__(self):
        return self.to_json_dict()

    def __str__(self):
        return self.to_json_dict()

    def __eq__(self, other):
        return self.to_sha256_str() == other.to_sha256_str()


class Config(NestedData):
    OPTIONAL_FIELDS = ()

    def __init__(self,
                 *args,
                 **kwargs):
        for field in self.FIELDS:
            if hasattr(self, field):
                if (getattr(self, field) is None) and (field not in self.OPTIONAL_FIELDS):
                    raise RuntimeError(f'{self.__class__}: the value for the field {field} '
                                       f'was not provided neither during initialisation, nor by default.\n')

        super().__init__(*args, **kwargs)


class PatternConfig(Config):
    ALLOWED_PATTERN_TYPES = ('uniform',)
    FIELDS = (
        'pattern_type',
    )
    UNIFORM_PATTERN_FIELD = None

    def __init__(self,
                 *args,
                 pattern_type: str = 'uniform',
                 **kwargs):
        self.pattern_type = pattern_type

        super().__init__(*args, **kwargs)

    def create_uniform_pattern(self,
                               depth: int = None):
        return (getattr(self, self.UNIFORM_PATTERN_FIELD),) * depth

    def create_non_uniform_pattern(self,
                                   depth: int = None):
        raise NotImplementedError

    def create_pattern(self, depth):
        assert self.pattern_type in self.ALLOWED_PATTERN_TYPES
        if self.pattern_type == 'uniform':
            return self.create_uniform_pattern(depth=depth)
        else:
            return self.create_non_uniform_pattern(depth=depth)


class Schedule:
    def __init__(self,
                 *args,
                 schedule: Tuple[Tuple[int, Config]] = None,
                 **kwargs):
        assert isinstance(schedule, tuple)
        for schedule_stage in schedule:
            assert isinstance(schedule_stage[0], int)
            assert issubclass(type(schedule_stage[1]), Config)

        self.schedule = schedule

    def __getitem__(self, key):
        return self.schedule[key]

    def __len__(self):
        return len(self.schedule)

    def to_dict(self):
        return {
            'schedule': tuple((schedule_stage[0], schedule_stage[1].to_dict()) for schedule_stage in self.schedule)
        }

    def to_json_dict(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_json(self, filename: str = None):
        with open(filename, 'w') as f:
            json.dump(self.to_json_dict(), f)

    def __hash__(self):
        return hash(self.to_json_dict())

    def __repr__(self):
        return self.to_json_dict()

    def __str__(self):
        return self.to_json_dict()

    def __eq__(self, other):
        return self.to_json_dict() == other.to_json_dict()




