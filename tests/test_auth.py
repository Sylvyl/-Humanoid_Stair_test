from x2_stair_autonomy.auth import SessionManager
import unittest


class AuthTests(unittest.TestCase):
    def test_login_and_csrf(self):
        manager = SessionManager("correct-horse-battery-staple")
        self.assertIsNone(manager.login("wrong"))
        session, csrf = manager.login("correct-horse-battery-staple")
        self.assertTrue(manager.validate(session, csrf))
        self.assertFalse(manager.validate(session, "wrong-csrf"))
