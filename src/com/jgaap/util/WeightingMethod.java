package com.jgaap.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.log4j.Logger;
import edu.stanford.nlp.util.ConfusionMatrix;
//import weka.classifiers.evaluation.ConfusionMatrix;

import com.jgaap.generics.AnalysisDriver;
import com.jgaap.generics.AnalyzeException;
/**
 * @author Alejandro J Napolitano Jawerbaum
 * This class provides support for weightedVoting and other algorithms that put AnalysisDrivers to a vote by weighting said votes.
 */
public class WeightingMethod {
	private static Logger logger = Logger.getLogger(WeightingMethod.class);
	private static Set<String> authorSet = new HashSet<String>();
	/**
	 * @param Set<AnalysisDriver> classifiers. A set of AnalysisDrivers.
	 * @param Set<Document> knownDocuments. These will be used for cross-validation
	 * @param String method. Name of the weighted algorithm.
	 * @param String authors. The authors to cross-validate for. An empty string means all authors will be cross-validated.
	 * @return Set<Pair<AnalysisDriver,Double>> So as to prevent having duplicate AnalysisDrivers.
	 * This algorithm simply takes in instructions and passes them onto the appropriate weighting method.
	 * Doing this because, in the case of suspected and distractor authors, a user may wish to only take into consideration how accurate an algorithm is at differentiating the suspected authors from each other and the distractor authors.
	 */
	public static Set<Pair<AnalysisDriver,Double>> weight(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String method, String authorsToCrossval, Set<String> authorS) throws AnalyzeException{
		 authorSet = authorS;
		 if(method.equalsIgnoreCase("Macro average precision"))
			 return macroAveragePrecision(classifiers, knownDocuments, authorsToCrossval);
		 else if(method.equalsIgnoreCase("Macro average recall"))
			 return macroAverageRecall(classifiers, knownDocuments, authorsToCrossval);
		 else if(method.equalsIgnoreCase("Balanced Accuracy Weighted"))
			 return balancedAccuracyWeighted(classifiers, knownDocuments, authorsToCrossval);
		 else if(method.equalsIgnoreCase("F1"))
			 return F1(classifiers, knownDocuments, authorsToCrossval);
		 else if(method.equalsIgnoreCase("Macro F1 of averages"))
			 return macroF1OfAverages(classifiers, knownDocuments, authorsToCrossval);
		 else if(method.equalsIgnoreCase("Macro Averaged F1"))
			 return averageMacroF1(classifiers, knownDocuments, authorsToCrossval);
		 else if(method.equalsIgnoreCase("Micro Averaged F1"))
			 return microAveragedF1(classifiers, knownDocuments, authorsToCrossval);
		 else
		 {
			 Set<Pair<AnalysisDriver, Double>> unweightedClassifiers = new HashSet<Pair<AnalysisDriver,Double>>(); 
			 for(AnalysisDriver classifier : classifiers)
				 unweightedClassifiers.add(new Pair<AnalysisDriver,Double>(classifier, 1.0));
			 return unweightedClassifiers;
		 }
	}
	
	/**
	 * This algorithm computes the confusion matrix of a classifier for several weighting methods.
	 * @throws AnalyzeException 
	 * 
	 */
	public static List<Pair<AnalysisDriver,ConfusionMatrix<String>>> calculateConfusionMatrices(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = new ArrayList<Pair<AnalysisDriver,ConfusionMatrix<String>>>();
		for(AnalysisDriver classifier : classifiers) {
			ConfusionMatrix<String> confusionMatrix = new ConfusionMatrix<String>();
			for (Document knownDocument : knownDocuments) {
				List<Document> knownDocuments2 = new ArrayList<Document>();
				if(authors.contains(knownDocument.getAuthor()) || authors.trim().isEmpty()){
					for(Document knownDocument2 : knownDocuments) {
						if(!knownDocument2.equals(knownDocument))
							knownDocuments2.add(knownDocument2);
					}
						logger.info("Training " + classifier.displayName() +" for cross-validation");
						classifier.train(knownDocuments2);
						logger.info("Finished Training "+classifier.displayName() + " for cross-validation");
						logger.info("Begining Analyzing: " + knownDocument.toString() + " for cross-validation");
						List<Pair<String,Double>> results = classifier.analyze(knownDocument);
						logger.info("Finished Analyzing: "+ knownDocument.toString() + " for cross-validation");
						confusionMatrix.add(results.get(0).getFirst(),knownDocument.getAuthor());
					}
			}
			confusionMatrices.add(new Pair<AnalysisDriver,ConfusionMatrix<String>>(classifier, confusionMatrix));
		}
		for(Pair<AnalysisDriver, ConfusionMatrix<String>> matrix : confusionMatrices)
			logger.info(matrix.getSecond().toString());
		return(confusionMatrices);
	}
	
	public static ConfusionMatrix<String> calculateConfusionMatrix(AnalysisDriver classifier, List<Document> knownDocuments) throws AnalyzeException{
		ConfusionMatrix<String> confusionMatrix = new ConfusionMatrix<String>();
		for (int i = 0; i<knownDocuments.size(); i++)
			confusionMatrix.add(knownDocuments.get(i).getFormattedResult(classifier),knownDocuments.get(i).getAuthor());
		return confusionMatrix;
	}
	/**
	 * @param Set<AnalysisDriver> classifiers. A set of AnalysisDrivers.
	 * @param Set<Document> knownDocuments. These will be used for cross-validation
	 * @param String authors. The authors to cross-validate for. An empty string means all authors will be cross-validated.
	 * This algorithm weights by raw LOOCV score.
	 */
	public static Set<Pair<AnalysisDriver, Double>> macroAveragePrecision(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double weight = 0.0;
			if(!authors.trim().isEmpty()) {
			for(String author : authors.split(",")) {
				weight += confusionMatrix.getSecond().getContingency(author.trim()).precision();
			}
			weight /= authors.split(",").length;
			weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
			else{
				for(String author : authorSet) {
					weight += confusionMatrix.getSecond().getContingency(author.trim()).precision();
				}
				weight /= authorSet.size();
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
		}
		return weights;	
	}
	public static Set<Pair<AnalysisDriver, Double>> weightedAveragePrecision(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double weight = 0.0;
			if(!authors.trim().isEmpty()) {
			for(String author : authors.split(",")) {
				weight += confusionMatrix.getSecond().getContingency(author.trim()).precision();
			}
			weight /= authors.split(",").length;
			weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
			else{
				for(String author : authorSet) {
					weight += confusionMatrix.getSecond().getContingency(author.trim()).precision();
				}
				weight /= authorSet.size();
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
		}
		return weights;	
	}
	/**
	 * @param Set<AnalysisDriver> classifiers. A set of AnalysisDrivers.
	 * @param Set<Document> knownDocuments. These will be used for cross-validation
	 * @param String authors. The authors to cross-validate for. An empty string means all authors will be cross-validated.
	 * This algorithm weights by raw LOOCV score divided by the total sum of weights.
	 */
	public static Set<Pair<AnalysisDriver, Double>> macroAverageRecall(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double weight = 0.0;
			if(!authors.trim().isEmpty()) {
			for(String author : authors.split(",")) {
				weight += confusionMatrix.getSecond().getContingency(author.trim()).recall();
			}
			weight /= authors.split(",").length;
			weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
			else{
				for(String author : authorSet) {
					weight += confusionMatrix.getSecond().getContingency(author.trim()).recall();
				}
				weight /= authorSet.size();
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
		}
		return weights;	
	}
	public static Set<Pair<AnalysisDriver, Double>> balancedAccuracyWeighted(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double weightedRecallSum = 0.0;
			Double sumWR = 0.0;
			if(!authors.trim().isEmpty()) {
				for(String author : authors.split(",")) {
					weightedRecallSum += confusionMatrix.getSecond().getContingency(author.trim()).recall()*(Collections.frequency(knownDocuments, author.trim())/knownDocuments.size());
					sumWR+= (Collections.frequency(knownDocuments, author.trim())/knownDocuments.size());
				}
				Double weight = weightedRecallSum/(authors.split(",").length*sumWR);
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
			else {
				for(String author : authorSet) {
					weightedRecallSum += confusionMatrix.getSecond().getContingency(author.trim()).recall()*(Collections.frequency(knownDocuments, author.trim())/knownDocuments.size());
					sumWR+= (Collections.frequency(knownDocuments, author.trim())/knownDocuments.size());
				}
				Double weight = weightedRecallSum/(authorSet.size()*sumWR);
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), weight));
			}
		}
		return weights;	
	}
	public static Set<Pair<AnalysisDriver, Double>> F1(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			if(!authors.trim().isEmpty())
				for(String author : authors.split(",")) 
					weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), confusionMatrix.getSecond().getContingency(author.trim()).f1()));
		else
			for(String author : authorSet)
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), confusionMatrix.getSecond().getContingency(author.trim()).f1()));
		}
		return weights;
	}
	/**
	 * Harmonic mean over Arithmetic means, as described by  Optiz and Burst (2021) https://arxiv.org/pdf/1911.03347.pdf
	 * @param classifiers
	 * @param knownDocuments
	 * @param authors
	 * @return
	 * @throws AnalyzeException
	 */
	public static Set<Pair<AnalysisDriver, Double>> macroF1OfAverages(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double macroAveragePrecision = 0.0;
			Double macroAverageRecall = 0.0;
			if(!authors.trim().isEmpty()) {
				for(String author : authors.split(",")) {
					macroAveragePrecision += confusionMatrix.getSecond().getContingency(author.trim()).precision();
					macroAverageRecall += confusionMatrix.getSecond().getContingency(author.trim()).recall();
				}
				macroAverageRecall /= authors.split(",").length;
				macroAveragePrecision /= authors.split(",").length;
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), 2*(macroAveragePrecision*macroAverageRecall)/((1/macroAveragePrecision)+(1/macroAverageRecall))));
			}
			else {
				for(String author : authorSet) {
					macroAveragePrecision += confusionMatrix.getSecond().getContingency(author.trim()).precision();
					macroAverageRecall += confusionMatrix.getSecond().getContingency(author.trim()).recall();
				}
				macroAverageRecall /= authorSet.size();
				macroAveragePrecision /= authorSet.size();
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), 2*(macroAveragePrecision*macroAverageRecall)/(macroAveragePrecision+macroAverageRecall)));
			}
		}
		return weights;	
	}
	/**
	 * Arithmetic mean over harmonic means, as described by  Optiz and Burst (2021) https://arxiv.org/pdf/1911.03347.pdf
	 * @param classifiers
	 * @param knownDocuments
	 * @param authors
	 * @return
	 * @throws AnalyzeException
	 */
	public static Set<Pair<AnalysisDriver, Double>> averageMacroF1(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double harmonicMeans = 0.0;
			if(!authors.trim().isEmpty()) {
				for(String author : authors.split(",")) 
					harmonicMeans+= 2*(confusionMatrix.getSecond().getContingency(author.trim()).precision()*confusionMatrix.getSecond().getContingency(author.trim()).recall())/(confusionMatrix.getSecond().getContingency(author.trim()).precision()+confusionMatrix.getSecond().getContingency(author.trim()).recall());				
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), harmonicMeans/authors.split(",").length));
			}
			else {
				for(String author : authorSet)
					harmonicMeans+= 2*(confusionMatrix.getSecond().getContingency(author.trim()).precision()*confusionMatrix.getSecond().getContingency(author.trim()).recall())/(confusionMatrix.getSecond().getContingency(author.trim()).precision()+confusionMatrix.getSecond().getContingency(author.trim()).recall());
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), harmonicMeans/authorSet.size()));
			}
		}
		return weights;	
	}
	public static Set<Pair<AnalysisDriver, Double>> microAveragedF1(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double TP = 0.0;
			Double rest = 0.0;
			Double F1 = 0.0;
			if(!authors.trim().isEmpty()) {
				for(String author : authors.split(",")) {
						TP += confusionMatrix.getSecond().get(author, author);
						for(String author2 : authors.split(","))
							if(author2 != author) {
								rest += confusionMatrix.getSecond().get(author, author2);
							}
				}
				F1 = TP/(TP+rest/2);
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), F1));
			}
			else {
				for(String author : authorSet) {
					TP += confusionMatrix.getSecond().get(author, author);
					for(String author2 : authorSet)
						if(author2 != author) {
							rest += confusionMatrix.getSecond().get(author, author2);
							}
				}
				F1 = TP/(TP+rest/2);
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(),F1));
			}
		}
		return weights;
	}

	/*
	public static Set<Pair<AnalysisDriver, Double>> mattheusCorrelationCoefficient(Set<AnalysisDriver> classifiers, List<Document> knownDocuments, String authors) throws AnalyzeException{
		List<Pair<AnalysisDriver,ConfusionMatrix<String>>> confusionMatrices = calculateConfusionMatrices(classifiers, knownDocuments, authors);
		Set<Pair<AnalysisDriver, Double>> weights = new HashSet<Pair<AnalysisDriver,Double>>();
		for(Pair<AnalysisDriver,ConfusionMatrix<String>> confusionMatrix : confusionMatrices) {
			Double c = 0.0;
			Double s = 0.0;
			if(!authors.isBlank()) {
				for(String author : authors.split(",")) {
				Double p = confusionMatrix.getSecond().
				}
				
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(), );
			}
			else {
				for(String author : authorSet) {
					
				}
				
				weights.add(new Pair<AnalysisDriver,Double>(confusionMatrix.getFirst(),);
			}
		}
		return weights;	
	}
	*/
}
