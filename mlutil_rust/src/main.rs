use std::env;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_clean_punctuation() {
        let s = String::from("def abc(x)");
        let s_clean = String::from("def abc x ");
        assert_eq!(remove_punctuation(&s, false), s_clean)
    }

    #[test]
    fn test_clean_multiple_whitespaces() {
        let s = String::from("a b");
        let s_other = String::from("a  b");
        assert_eq!(remove_punctuation(&s, false), remove_punctuation(&s_other, false))
    }

    #[test]
    fn test_tokenize() {
        let s = String::from("def abc(x)");
        let s_clean: Vec<String> = ["def", "abc", "x"].map(String::from).to_vec();
        let tokens = tokenize(&s, false);
        for (token, expected) in tokens.iter().zip(s_clean.iter()) {
            assert_eq!(token, expected)
        }
    }
}

fn is_punctuation(ch: char, remove_underscore: bool) -> bool {
    const CHS: &'static [char] = &['(', ')', ',', '\"', '.', ';', ':', '\''];
    let is_in_punctuation = CHS.contains(&ch);
    if remove_underscore {
        (ch == ' ') | is_in_punctuation
    } else {
        is_in_punctuation
    }
}

fn remove_punctuation(s: &String, remove_underscore: bool) -> String {
    const WHITESPACE: char = ' ';
    s.chars()
        .map(|x| match is_punctuation(x, remove_underscore) {
            true => WHITESPACE,
            false => x,
        })
        .collect()
}

fn tokenize(s: &String, remove_underscore: bool) -> Vec<String> {
    let cleaned_str = remove_punctuation(&s, remove_underscore);
    let splits = cleaned_str.split_whitespace();
    splits.map(String::from).collect()
}

fn tokenize_camelcase(s: &String) -> &String {
    s
}

fn print_nums() {
    let nums = [1, 2, 3];
    let filtered_nums = nums.iter().filter(|&&i| i > 2);
    for i in filtered_nums {
        println!("{}", i);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let remove_underscore = true;
    let arg = &args[2];
    println!("original: str {}", &arg.clone());
    let tokens = tokenize(&arg, remove_underscore);
    for (i, token) in tokens.iter().enumerate() {
        println!("token {}: {}", i, token);
    }
}
