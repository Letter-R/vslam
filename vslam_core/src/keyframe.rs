use image::DynamicImage;
pub struct KeyFrame {
    id: usize,                       // 编号
    image_path: String,              // 文件地址
    features: Vec<Box<dyn Feature>>, // 特征
}

pub trait Feature {
    fn extract_features(image: &DynamicImage) -> Vec<Self>
    where
        Self: Sized;
    fn match_features(feature1: &Vec<Self>, feature2: &Vec<Self>) -> Vec<(usize, usize)>
    where
        Self: Sized;
}
