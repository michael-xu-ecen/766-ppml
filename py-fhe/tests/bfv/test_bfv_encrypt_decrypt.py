import os
import unittest
import re
from bfv.bfv_decryptor import BFVDecryptor
from bfv.bfv_encryptor import BFVEncryptor
from bfv.bfv_key_generator import BFVKeyGenerator
from bfv.bfv_parameters import BFVParameters
from util.plaintext import Plaintext
from util.polynomial import Polynomial
from util.random_sample import sample_uniform

TEST_DIRECTORY = os.path.dirname(__file__)


def extract_and_sort_coefficients(decrypted_message_string):
    # Split the string by whitespace and strip numbers before each "x"
    coefficients_str = [s.split('x')[0] for s in decrypted_message_string.strip().split()]
    coefficients_str = [i for i in coefficients_str if i != '+']
    coefficients_str = [1 if x == '' else x for x in coefficients_str] 
    coefficients_str.reverse()

    return coefficients_str


def encrypt_numbers_in_file(input_file_path, output_file_path):
    params = BFVParameters(poly_degree=2048,  # Set your parameters accordingly
                           plain_modulus=256,
                           ciph_modulus=0x3fffffff000001)
    key_generator = BFVKeyGenerator(params)
    public_key = key_generator.public_key
    encryptor = BFVEncryptor(params, public_key)

    with open(input_file_path, 'r') as input_file:
        content = input_file.read()
    
    # Extract only the numbers using regular expression
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', content)

    # Encrypt each number
    encrypted_numbers = []
    for num in numbers:
        encrypted_number = encryptor.encrypt(Plaintext(Polynomial(1, num))) 
        encrypted_numbers.append(str(encrypted_number))

    # Write encrypted numbers to output file
    with open(output_file_path, 'w') as output_file:
        output_file.write('\n'.join(encrypted_numbers))


class TestEncryptDecrypt(unittest.TestCase):
    def setUp(self):
        print("Setting up test...")
        self.small_degree = 5
        self.small_plain_modulus = 60
        self.small_ciph_modulus = 50000
        self.large_degree = 2048
        self.large_plain_modulus = 256
        self.large_ciph_modulus = 0x3fffffff000001

    def run_test_tiny_encrypt_decrypt(self, message):
        print("Running tiny encrypt-decrypt test...")
        params = BFVParameters(poly_degree=self.small_degree,
                               plain_modulus=self.small_plain_modulus,
                               ciph_modulus=self.small_ciph_modulus)
        key_generator = BFVKeyGenerator(params)
        public_key = key_generator.public_key
        secret_key = key_generator.secret_key
        encryptor = BFVEncryptor(params, public_key)
        decryptor = BFVDecryptor(params, secret_key)
        message = Plaintext(Polynomial(self.small_degree, message))
        ciphertext = encryptor.encrypt(message)
        decrypted_message = decryptor.decrypt(ciphertext)
        
        decrypted_message_string = str(decrypted_message)
        # Extract and sort coefficients
        coefficients = extract_and_sort_coefficients(decrypted_message_string)
        print("Sorted coefficients:", coefficients)
        
        self.assertEqual(str(message), decrypted_message_string)

    def run_test_large_encrypt_decrypt(self, message):
        print("Running large encrypt-decrypt test...")
        params = BFVParameters(poly_degree=self.large_degree,
                               plain_modulus=self.large_plain_modulus,
                               ciph_modulus=self.large_ciph_modulus)
        key_generator = BFVKeyGenerator(params)
        public_key = key_generator.public_key
        secret_key = key_generator.secret_key
        encryptor = BFVEncryptor(params, public_key)
        decryptor = BFVDecryptor(params, secret_key)
        message = Plaintext(Polynomial(self.large_degree, message))
        ciphertext = encryptor.encrypt(message)
        decrypted_message = decryptor.decrypt(ciphertext)
        
        decrypted_message_string = str(decrypted_message)
        
        # Extract and sort coefficients
        coefficients = extract_and_sort_coefficients(decrypted_message_string)
        #print("Sorted coefficients:", coefficients)
        
        self.assertEqual(str(message), decrypted_message_string)

    def test_tiny_encrypt_decrypt_01(self):
        print("Running test_tiny_encrypt_decrypt_01...")
        array = [0, 23, 48, 52, 6]
        # Fix the zero later
        self.run_test_tiny_encrypt_decrypt(array)

    def test_tiny_encrypt_decrypt_02(self):
        print("Running test_tiny_encrypt_decrypt_02...")
        self.run_test_tiny_encrypt_decrypt([19, 7, 42, 1, 54])

    def test_large_encrypt_decrypt_01(self):
        print("Running test_large_encrypt_decrypt_01...")
        vec = sample_uniform(0, self.large_plain_modulus, self.large_degree)
        #print(vec)
        self.run_test_large_encrypt_decrypt(vec)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)

    input_file_name = 'BC-TCGA-Normal.txt'  # File name of your input file
    output_file_name = 'encrypted_values.txt'  # File name of output encrypted file
    input_file_path = os.path.join(TEST_DIRECTORY, input_file_name)
    output_file_path = os.path.join(TEST_DIRECTORY, output_file_name)
    encrypt_numbers_in_file(input_file_path, output_file_path)
