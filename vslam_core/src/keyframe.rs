use image::DynamicImage;
pub struct KeyFrame{
    id:usize,
    image_path:String,
    features:Vec<Box<dyn Feature>>,
}


pub trait Feature {
    fn extract_features(image:&DynamicImage)->Vec<Self> where Self:Sized;
    fn match_features(feature1:Vec<Self>,feature2:Vec<Self>)->Vec<(usize,usize)> where Self:Sized;
}
