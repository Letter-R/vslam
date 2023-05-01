use image::DynamicImage;
use nalgebra::Vector2;
trait Feature {
    fn extract_features(image:&DynamicImage)->Vec<Self> where Self:Sized;
    fn match_features(feature1:Vec<Self>,feature2:Vec<Self>)->Vec<(usize,usize)> where Self:Sized;
}

pub struct ORBFeature{
    location:Vector2<f64>,
    descriptor:[u64;4],
}

impl Feature for ORBFeature {
    fn extract_features(image:&DynamicImage)->Vec<Self> where Self:Sized {
        todo!()
    }

    fn match_features(feature1:Vec<Self>,feature2:Vec<Self>)->Vec<(usize,usize)> where Self:Sized {
        todo!()
    } 

}