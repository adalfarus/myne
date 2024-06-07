import os
from enum import Enum
import re
from typing import Union, Tuple, Literal


class CryptMode(Enum):
    ENCRYPT = 'e'
    DECRYPT = 'd'


class Crypt:
    @staticmethod
    def a1z26(plaintext: Union[str, list], mode: CryptMode = CryptMode.ENCRYPT) -> list:
        if mode == CryptMode.ENCRYPT:
            return [(ord(x.lower()) - 96) if x.isalpha() else int() for x in plaintext]
        elif mode == CryptMode.DECRYPT:
            l = []
            for x in plaintext:
                if (isinstance(x, int) and x != 0) or (isinstance(x, str) and x.isdigit() and int(x) != 0):
                    l.append(chr(int(x) + 96))
            return l

    @staticmethod
    def _rot155(plaintext: str, random_array: list = None) -> list:
        if random_array is None:
            random_array = [x + ord(os.urandom(1)) for x in range(len(plaintext))]
        if isinstance(plaintext, str):
            return [ord(plaintext[x]) - random_array[x] for x in range(len(plaintext))]
        else:
            return [plaintext[x] - random_array[x] for x in range(len(plaintext))]

    @staticmethod
    def _crack_case(ciphertext: list) -> list:
        result = []
        pattern = r"-?\d{1,3}"
        for part in ciphertext:
            matches = re.findall(pattern, part)
            result.extend(matches)
        return [int(match) for match in result]

    @staticmethod
    def _join_items_with_condition(items):
        result = []
        for x, y in zip(items[::2], items[1::2]):
            if ord(os.urandom(1)) % 2 == 0:
                result.append(f"{x}-{y}")
            else:
                result.append(f"{x}{y}")
        return result

    @classmethod
    def rot_away115(cls, plaintext: Union[str, list], random_array_overwrite: list = None,
                    mode: CryptMode = CryptMode.ENCRYPT) -> Tuple[list, list]:
        if isinstance(plaintext, str) and len(plaintext) % 2 != 0:
            plaintext += " "
        if random_array_overwrite is None:
            random_array_overwrite = []

        if mode == CryptMode.ENCRYPT:
            while True:
                random_array = random_array_overwrite or [x + ord(os.urandom(1)) for x in range(len(plaintext))]
                encrypted = cls._rot155(plaintext, random_array)
                double_encrypted = cls._join_items_with_condition(encrypted)
                rotten = cls._rot155(plaintext)
                double_rotten = cls._join_items_with_condition(rotten)
                encrypted2 = [int(x) for x in cls._crack_case(double_encrypted)]
                negated_random_array = [(x * -1) for x in random_array]
                plainnumbers = cls._rot155(encrypted2, negated_random_array)
                decrypted = [chr(x) if x > 0 and x < 126 else x for x in plainnumbers]
                decryptedtext = "".join([str(x) if not isinstance(x, str) else x for x in decrypted])
                if decryptedtext == plaintext:
                    return double_encrypted, random_array
        elif mode == CryptMode.DECRYPT:
            random_array = random_array_overwrite or [x + ord(os.urandom(1)) for x in range(len(plaintext))]
            encrypted_numbers = [int(x) for x in cls._crack_case(plaintext)]
            negated_random_array = [-x for x in random_array]
            plainnumbers = cls._rot155(encrypted_numbers, negated_random_array)
            decrypted = [chr(x) if 0 < x < 126 else x for x in plainnumbers]
            decrypted_text = ''.join([str(x) if not isinstance(x, str) else x for x in decrypted])
            return decrypted_text

    @staticmethod
    def caesar_ciper_adv(shift_value: int, plaintext: str,
                         char_string: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                         maintain_case: Literal["output", "char_string", None] = None,
                         missing_char_error: bool = True) -> str:
        strict_case_char_string = maintain_case == "char_string"
        maintain_case_in_output = maintain_case == "output"
        if not strict_case_char_string:
            seen = set()
            lower_char_string = []
            for c in char_string:
                lower_c = c.lower()
                if lower_c not in seen:
                    seen.add(lower_c)
                    lower_char_string.append(lower_c)
            char_string = ''.join(lower_char_string)

        char_index_map = {char: idx for idx, char in enumerate(char_string)}
        char_string_length = len(char_string)
        result = []

        for char in plaintext:
            case = char.isupper() and maintain_case_in_output
            if not strict_case_char_string:
                char = char.lower()

            if char in char_index_map:
                char_index = char_index_map[char]
                new_pos = (char_index + shift_value) % char_string_length
                new_char = char_string[new_pos].upper() if case else char_string[new_pos]
                result.append(new_char)
            else:
                if missing_char_error:
                    raise ValueError(f"Char '{char}' not in char_string '{char_string}'.")
                print(f"Char '{char}' not in char_string '{char_string}'.")
                result += char
        return ''.join(result)


if __name__ == "__main__":
    encrypted, random_array = Crypt.rot_away115(("HELLO WORLD" * 10_000), [], CryptMode.ENCRYPT)
    print("decrypted", Crypt.rot_away115(encrypted, random_array, CryptMode.DECRYPT))
    encrypted = Crypt.caesar_ciper_adv(7000000000000, "HEL:", missing_char_error=False)
    print(Crypt.caesar_ciper_adv(-7, encrypted, missing_char_error=False))


class CharSet(Enum):
    NUMERIC = "Numeric"
    ALPHA = "Alphabetic"
    ALPHANUMERIC = "Alphanumeric"
    ASCII = "ASCII"
    UNICODE = "Unicode"


class ComputerType(Enum):
    # Operations per second
    FASTEST_COMPUTER = 10**30 // 2  # One Quintillion (exascale), divided by 2 due to storage speed limits
    SUPERCOMPUTER = 1_102_000_000_000_000_000 // 2  # (Frontier), divided by 2 due to storage speed limits
    HIGH_END_PC = 1_000_000_000  # high-end desktop x86 processor
    NORMAL_PC = 50_000_000  # Example: A standard home computer
    OFFICE_PC = 1_000_000  # Example: An office PC
    OLD_XP_MACHINE = 10_000  # Example: An old Windows XP machine
    UNTHROTTLED_ONLINE = 10
    THROTTLED_ONLINE = 0.02777777777777778


class Efficiency(Enum):
    LOW = 0.5  # Low efficiency
    MEDIUM = 0.75  # Medium efficiency
    HIGH = 1.0  # High efficiency


class HashingAlgorithm(Enum):
    NTLM = 1_000_000_000  # NTLM (fast)
    MD5 = 500_000_000  # MD5 (medium speed)
    SHA1 = 200_000_000  # SHA1 (slow)
    BCRYPT = 1_000  # bcrypt (very slow)


class PasswordCrackEstimator:
    CHARSET_RANGES = {
        CharSet.NUMERIC: "0123456789",
        CharSet.ALPHA: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        CharSet.ALPHANUMERIC: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        CharSet.ASCII: ''.join(chr(i) for i in range(128)),
        CharSet.UNICODE: ''.join(chr(i) for i in range(1114112))
    }

    CHARSET_SIZES = {
        CharSet.NUMERIC: 10,
        CharSet.ALPHA: 26 * 2,
        CharSet.ALPHANUMERIC: 10 + 26 * 2,
        CharSet.ASCII: 128,
        CharSet.UNICODE: 1114112
    }

    def __init__(self, charset, computer_type, efficiency, hashing_algorithm):
        self.charset = charset
        self.computer_type = computer_type
        self.efficiency = efficiency
        self.hashing_algorithm = hashing_algorithm

    def estimate_time_to_crack(self, password: str, length_range: Optional[Union[int, tuple]] = None,
                               personal_info: Optional[list] = None):
        cleaned_password = self._remove_common_patterns(password, personal_info)
        if not self._filter_by_charset(password):
            return cleaned_password, "Password not in charset"

        if isinstance(length_range, int):
            length_range = (length_range, length_range)
        else:
            length_range = length_range

        if length_range and (len(cleaned_password) < length_range[0] or len(cleaned_password) > length_range[1]):
            return cleaned_password, "Password length is outside the specified range"

        charset_size = self.CHARSET_SIZES[self.charset]
        if length_range:
            min_length, max_length = length_range
            total_combinations = sum(charset_size ** length for length in range(min_length, max_length + 1))
        else:
            total_combinations = charset_size ** len(cleaned_password)

        adjusted_ops_per_second = self.computer_type.value * self.efficiency.value * self.hashing_algorithm.value
        estimated_seconds = total_combinations / adjusted_ops_per_second
        return cleaned_password, self._format_time(estimated_seconds)

    def _remove_common_patterns(self, password, personal_info):
        common_patterns = self._generate_common_patterns()
        for pattern in common_patterns:
            if pattern in password:
                password = password.replace(pattern, "")

        if personal_info:
            personal_info.extend(self._generate_birthday_formats(personal_info))
            for info in personal_info:
                if info in password:
                    password = password.replace(info, "")
                if self._is_significant_match(info, password):
                    password = ''.join([c for c in password if c not in info])

        return password

    def _filter_by_charset(self, password):
        valid_chars = self.CHARSET_RANGES[self.charset]
        for char in password:
            if char in valid_chars:
                continue
            return False
        return True

    def _generate_common_patterns(self):
        patterns = []
        for i in range(10000):  # Generate number sequences from 0000 to 9999
            patterns.append(str(i).zfill(4))
            patterns.append(str(i).zfill(4)[::-1])  # Add reversed patterns
        return patterns

    @staticmethod
    def _generate_birthday_formats(personal_info):
        formats = []
        for info in personal_info:
            if len(info) == 8 and info.isdigit():  # YYYYMMDD
                year, month, day = info[:4], info[4:6], info[6:8]
                formats.extend([
                    f"{year}{month}{day}",
                    f"{day}{month}{year}",
                    f"{month}{day}{year}",
                    f"{year}-{month}-{day}",
                    f"{day}-{month}-{year}",
                    f"{month}-{day}-{year}",
                    f"{month}-{day}",
                    f"{month}{day}",
                    f"{day}-{month}",
                    f"{day}{month}",
                    f"{year}-{day}",
                    f"{year}{day}",
                    f"{day}-{year}",
                    f"{day}{year}"
                    f"{year}-{month}",
                    f"{year}{month}",
                    f"{month}-{year}",
                    f"{month}{year}"
                ])
        return formats

    def _is_significant_match(self, info, password):
        matches = sum(1 for char in info if char in password)
        return matches / len(info) >= 0.9

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            return f"{seconds / 60:.2f} minutes"
        elif seconds < 86400:
            return f"{seconds / 3600:.2f} hours"
        elif seconds < 31536000:
            return f"{seconds / 86400:.2f} days"
        else:
            return f"{nice_number(int(seconds / 31536000))} years"


if __name__ == "__main__":
    charset = CharSet.ASCII
    computer_type = ComputerType.FASTEST_COMPUTER
    efficiency = Efficiency.HIGH
    hashing_algorithm = HashingAlgorithm.NTLM

    estimator = PasswordCrackEstimator(charset, computer_type, efficiency, hashing_algorithm)
    password = "HMBlw:_88008?@"
    personal_info = ["John", "19841201", "Doe"]
    actual_password, time_to_crack = estimator.estimate_time_to_crack(password, length_range=(1, 24), personal_info=personal_info)
    print(f"Estimated time to crack the password '{password}' [{actual_password}]: {time_to_crack}")
