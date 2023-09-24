from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import full_check, get_list

# The source data I will test on
true_data = [0.2683134097105212, 0.2848589191898987, 0.2656988172122933,
             0.18095763134156362, 0.17996607210955454, 0.17087615229095043,
             0.12964932745571078, 0.13317405614855718, 0.1261114052668434]



class Tests5(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed!")

        if reply.count('[') != 1 or reply.count(']') != 1:
            return CheckResult.wrong('No expected list was found in output!')

        # Getting the student's results from the reply

        try:
            student, _ = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that data output is in wrong format!')

        error = 'Incorrect silhouette scores.'
        check_result = full_check(student, true_data, '', tolerance=0.1, error_str=error)
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests5().run_tests()
