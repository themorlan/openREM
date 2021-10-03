# This Python file uses the following encoding: utf-8
# test_hash_id.py

from django.test import TestCase
from pydicom.multival import MultiValue
from pydicom.valuerep import MultiString, PersonName
from remapp.tools.hash_id import hash_id


class HashIDTests(TestCase):
    def test_hash_id_string(self):
        """
        Test an ID returns the expected hash
        """
        id = "1234567a"
        hashed_id = "47ef20207489b775fa4cdcac3c394b517ab22d7460237ae3df1ac0e8963699d6"
        self.assertEqual(hash_id(id), hashed_id)

    def test_hash_id_multivalue(self):
        """
        Test an ID returns the expected hash after converting from MultiVal
        """
        id = r"123\4567a"
        multivalue_id = MultiString(id)
        hashed_id = "47ef20207489b775fa4cdcac3c394b517ab22d7460237ae3df1ac0e8963699d6"
        self.assertEqual(hash_id(multivalue_id), hashed_id)

    def test_hash_id_differs(self):
        """
        Test passing two different IDs returns different hash values
        """
        id_1 = "1234567a"
        id_2 = "1234567b"
        self.assertNotEqual(hash_id(id_1), hash_id(id_2))

    def test_hash_non_ascii(self):
        """
        Test hash of non-ASCII values return the expected hash
        :return:
        """
        id = "123íä日本語文字列"
        hashed_id = "a74d459c48304dfdb56808558f783e1761eef18b4a59f5c1b3fef348b809f434"
        self.assertEqual(hash_id(id), hashed_id)

    def test_hash_name(self):
        """
        Test hash of PersonName
        :return:
        """
        name = PersonName("Fysiikka^kuvanlaatu")
        self.assertEqual(
            hash_id(name),
            "6a67e5fff39b58e1a6ab96373f263f391036b8a138c7c5e298a88dbb12416a49",
        )

    def test_multivalue_names(self):
        """
        Test hash of multivalued name. Shouldn't be used, but shouldn't cause error.
        :return:
        """
        multiname = MultiValue(PersonName, [PersonName("Müller"), PersonName("Smith")])
        self.assertEqual(
            hash_id(multiname),
            "b0b1b8027aecb0fef1831a0165deee00a77842d17409f3f3c8a186a6b5b4cda3",
        )
