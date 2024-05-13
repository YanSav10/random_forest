import ast
from hstest import CheckResult

def get_list(s):
    index_from = s.find('[')
    index_to = s.find(']')
    data_str = s[index_from: index_to + 1]
    data_list = ast.literal_eval(data_str)
    if index_to + 2 > len(s):
        return data_list, None
    else:
        return data_list, s[index_to + 2:]


def full_check(result: list, true_result: list, name: str, tolerance=0.05, error_str = None):
    # Checking if the student's reply is a list
    if not isinstance(result, list):
        return CheckResult.wrong(f'Output for {name} is not a list.')

    # Checking size of the student's reply
    if len(result) != len(true_result):
        return CheckResult.wrong(f'Output for {name} should contain {len(true_result)} values,\
        found {len(result)}.')

    # Checking values of the student's reply
    for value, true_value in zip(result, true_result):
        if tolerance:
            if not (abs((value - true_value) / true_value) < tolerance):
                if error_str is not None:
                    return CheckResult.wrong(error_str)
                else:
                    return CheckResult.wrong(f'Incorrect {name} values. Check your {name} function.')
        else:
            if value != true_value:
                if error_str is not None:
                    return CheckResult.wrong(error_str)
                else:
                    return CheckResult.wrong(f'Incorrect {name} values. Check your {name} function.')

    return None
