from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import full_check, get_list

# The source data I will test on
true_data = [0.2223761642306306, -0.45683192123043, -0.03553715904637069,
             -0.5205156147670967, 0.24228760800150498, 0.7490126751320859,
             0.8100848687459451, -0.7101826652172734, 0.6714050998519755,
             -0.12395354307645551, 0.5146091721832895, 0.7159734076939872,
             0.4880573819913283, 0.7695125729142503, 0.41684421833212637,
             0.29732648583839666, -0.06497338195919612, 0.5597907911714886,
             0.679666348234634, 0.6568971990793854, -0.10702784137414335,
             0.029712309527479542, -0.18486985441333234, 0.36844490251533485,
             0.5646143799906963, 0.7521079040204687, -0.3269594980324239,
             0.346477567433997, -0.01617060925721845, 0.48267261975886366,
             -0.31087795930835155, -0.7899804364603202, -0.8416648901080834,
             0.6615422984440146, -0.6137856990911499, 0.14251807350872314,
             -0.5267179542501306, -0.7412851990697756, -0.5650928657163976]


class Tests2(StageTest):

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

        error = 'Incorrect new cluster centers values. Check your calculate_new_centers function.'
        check_result = full_check(student, true_data, '', tolerance=0.1, error_str=error)
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests2().run_tests()
