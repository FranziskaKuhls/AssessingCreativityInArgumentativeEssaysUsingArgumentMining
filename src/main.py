import sys
import requests


def evaluate_essay_with_old_metric(essay: str):
    data = {'dataset': "climate_change", 'text': essay}
    r = requests.post('http://localhost:5000/creativity', json=data)
    r.raise_for_status()
    if r.status_code != 204:
        result_json = r.json()
        return result_json
    return None


def evaluate_essay_acl(essay: str):
    data = {'text': essay}
    r = requests.post('http://localhost:5001/creativity', json=data)
    r.raise_for_status()
    if r.status_code != 204:
        result_json = r.json()
        return result_json
    return None


def evaluate_essay_targer(essay: str, essay_tags):
    data = {'text': essay, 'tags': essay_tags}
    r = requests.post('http://localhost:5002/creativity', json=data)
    r.raise_for_status()
    if r.status_code != 204:
        result_json = r.json()
        return result_json
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Run paramters (essay, tagged essay information) not given.")
    text = sys.argv[1]
    tags = sys.argv[2]
    print("Old CST metric: ")
    print(evaluate_essay_with_old_metric(text), "\n")
    print("CST With ACL metric: ")
    print(evaluate_essay_acl(text), "\n")
    print("CST With Targer metric: ")
    print(evaluate_essay_targer(text, tags), "\n")
