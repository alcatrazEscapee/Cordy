use std::{env, fs};
use std::path::PathBuf;

use crate::SourceView;

pub struct Resource {
    root: PathBuf,
    path: &'static str
}

pub fn get_resource(resource_type: &'static str, path: &'static str) -> Resource {
    Resource {
        root: [env::var("CARGO_MANIFEST_DIR").unwrap(), String::from("resources"), String::from(resource_type), format!("{}.cor", path)].iter().collect::<PathBuf>(),
        path,
    }
}

impl Resource {

    pub fn view(self: &Self) -> SourceView {
        SourceView::new(format!("{}.cor", self.path), fs::read_to_string(&self.root).expect(format!("Reading: {:?}", self.root).as_str()))
    }

    /// Takes `actual`, writes it to `.cor.out`, and compares it against the `.cor.trace` file
    pub fn compare(self: &Self, actual: Vec<String>) {
        let actual: String = actual.join("\n");
        let expected: String = fs::read_to_string(self.root.with_extension("cor.trace"))
            .expect(format!("Reading: {:?}", self.root).as_str());

        fs::write(self.root.with_extension("cor.out"), &actual).unwrap();

        assert_eq!(actual, expected.replace("\r", ""));
    }
}