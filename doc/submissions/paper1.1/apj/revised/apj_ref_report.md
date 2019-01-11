## Referee Report 

#### Reviewer's Comments:
The paper 'IQ-Collaboratory 1.1: the Star-Forming Sequence of Simulated Central
Galaxies' by Hahn et al. compares the star-forming sequence (SFS) between
various simulations and SDSS. The key aspect of the comparison is the usage of
Gaussian Mixture Models to identify the SFS. The paper is written well, and the
results are definitely interesting and important for the community. I therefore
recommend this paper after some moderate revision (see below). 

#### Major comments:
---

> The authors discuss in Appendix C resolution questions regarding the SFR
estimates. However, also the stellar masses themselves will not be fully
converged for different resolution levels of the different simulations. How can
the authors compare stellar mass estimates from simulations with different mass
resolution? For example, a given simulation (e.g., EAGLE) will lead to different
stellar masses depending on the mass resolution. The same is true for the other
hydro simulations. So, how can the stellar mass of a galaxy be compared
between two simulations with different mass resolutions?

We will more clearly mention in Section 2 that we impose a 100 star particle lower limit so stellar masses are not significantly impacted by the resolution. 

> How sensitive is the identification of the SFS to the 0.5 dex value for the
difference between the mean values? Can the authors please demonstrate that
this value does not affect the SFS identification in any significant way?

I’ve already tested the SFS identification with a range of values. We will including a description of these tests in the text of Section3.

> I understand the advantages of the Gaussian Mixture Models. However, can the
authors please clarify a few things: 
-Why shall we assume that the distribution of the different populations in the
Mstar - SFR plane are indeed Gaussian? How good is this approximation? Can the
authors please quantify non-Gaussian contributions?
-If k=3 is a good fit (based on the BIC), what does it physically mean to have
three components? The authors give some lose definition like low-SF, med-SF,
high-SF. But how certain can we be sure that these are real different
subpopulations?

We will clarify that fitting the total distribution GMMs does 
not assume that the populations are Gaussian. Instead, the 
fact that the BIC is lower for GMMs with 2 or 3 components
indicate that the subpopulations are well described by 
Gaussians. From a data standpoint, the real subpopulations 
are not well defined. We therefore avoid defining the populations
beyond low, med, and high-SF.

> The authors compare SDSS centrals with simulation centrals. However, they
measure the stellar mass of simulated galaxies using all star particles
belonging to halos. It has been demonstrated multiple times in the past, that
this will overestimate stellar masses significantly, especially towards the
massive/bright end. I am therefore wondering how the SFS, which is a function
of stellar mass, can be compared meaningfully between simulation and SDSS, if
the the stellar masses are not handled equally.

We will further emphasize the caveats of the comparison
and mention that this will be explicitly addressed in the 
next paper. 

>I understand that the authors mainly focused on the z=0 analysis in their
work. However, it would be very useful to see how these simulations compare at
higher z. Can the authors please repeat their analysis for one or two more
redshifts and add these results? Do the different simulations predict different
redshift behavior for the best-fit SFS? I feel that a paper that is dedicated
to an SFS comparison should at least briefly also discuss some higher z
aspects. 

We will mention in Section 4 that this is a paper in 
preparation. 

> The authors find that the different simulations indeed show rather large
difference once analyzed within a common framework (GMM). They also state that
they can not go into the detailed causes for these differences in the paper,
what I understand given the complexity of such an endeavor. However, one
question could potentially be tackled: the SFS is a relation between stellar
masses and SFR. Can the authors try to disentangle whether the discrepancy
between the different simulations is caused by issues with the stellar masses
or more related to the actual star formation rates? For that it would be
interesting to plot, for example, the star formation rate and stellar masses
vs. halo mass. Can the authors try to gain some insights into this?

See above

#### Minor comments:
----

> Regarding the comparison in Appendix A: Can the authors add another panel to
Fig. A1 where they show all simulations with no cut. How large are the
differences then? Also, it seems that mainly the amplitude changes in the
current right panel, but the slopes still seem to be consistent. Can the
authors please mention this and comment on it? Why is the slope such a stable prediction
in that case?

We will add another panel to the figure. 

> Have the author tested measuring galactic properties (SFR and stellar masses)
within radial cuts? How does this impact the SFS comparison?

> Can the authors please add some more background information about the Bayesian
Information Criteria? This seems to be some crucial step to pick out the right
k. So I wonder how certain this step is. What if it does not pick out the
right k, would it miss the SFS? 

We will include some extra background information 
on the BIC. We will also emphasize that SFS is defined 
as the dominant subpopulation in the SF portion of the 
distribution. So different k values mainly result in 
slight changes in the mu and sigma of the SFS 
Gaussian.

> Genel et al. recently had a paper on the the scatter of scaling relations
being potentially be driven by chaotic effects. Can the authors comment on how
sensitive the sigma comparison is to this?

> As the authors state the scatter in the SFS is lower than the observational
~0.3 value. The authors give some possible explanation for this, like
observational errors and missing burstiness of the simulation results. Can the
authors try to quantify those roughly and give some rough numbers? For example,
Sparre performed some re-simulations of Illustris galaxies and found more bursty
behavior with higher resolution. 

Chris, do you have any comments on this? We will
also include uncertaints on the SFR from the 
MPA-JHU catalog. 

> For the calculation of the instant. SFR: Which gas is considered for this? All
gas gravitationally bound to the halo?

We will mention in Section 2 that all gas gravitationally 
bound to the subhalo are included in the instantaneous
SFR.

> It would be nice to add a simulation table to the paper summarizing the key
numerical parameters (e.g., mass resolution, softening, etc.). 

We will include a table summarizing the key parameters 
(completeness of group finder, cosmic SFR density, etc.)

> The draft currently lacks any more detailed discussion of the discrepancies found
for the SFS fits. It would be nice to add least add a short discussion trying to shed
a bit light on the question why different simulations predict different SFS fits
and why those differ from SDSS.

See above


> Regarding the third bullet point of the conclusions: Based on Fig. 9 I would
think that Illustirs shows at least some stellar mass dependence of the
quiescent fraction and also has (similar to the SC-SAM and SDSS) some small
contribution of low mass star forming galaxies. That should be
mentioned here.

We will include this in the conclusions. 
