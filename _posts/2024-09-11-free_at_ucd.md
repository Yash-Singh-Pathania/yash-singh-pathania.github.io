---
title: "Free At UCD ? Free Food for Students with Real-Time Availability"
date: 2024-09-11
permalink: /posts/on_my_campus
categories: [UCD, Technology, Student Life]
tags:
  - python
  - fastApi
  - SelfProject
  - Streamlit
  - PostgreSQL
  - Render
mermaid: true
author: Yash Singh Pathania
---

*[Free at UCD](https://www.freeatucd.com)*
 is a casual, real-time platform created to help students find free food on campus! University life is packed with events, and this app helps you stay in the know about food giveaways, helping students save on costs and reducing food waste.

--- 


*Free at UCD* is just a bunch of kids coming together and code  a  project aimed at helping University College Dublin (UCD) students find free food giveaways across the campus. Launched just a day ago, the platform has already garnered over **100 active users**, demonstrating the significant demand and utility of such a service among students.

![Free at UCD Banner](/images/Freeatucd.png)

## The Inspiration Behind Free at UCD

University life comes with its own set of challenges, and managing expenses is high on that list. Recognizing the frequency of events offering free food and the lack of a centralized platform to inform students about them, we developed *Free at UCD*. Our goal is to reduce food waste and help students save money by connecting them to these opportunities in real-time.

## Technical Overview

Building an application that could handle real-time data and scale efficiently was crucial. Here's how we achieved it:

### Streamlit for Rapid Development

We chose [Streamlit](https://streamlit.io/) to build our Minimum Viable Product (MVP) due to its simplicity and the speed at which we could develop interactive web applications. Streamlit allowed us to focus on core functionalities without worrying about the complexities of traditional web development frameworks.

### PostgreSQL Database

For our database, we utilized [PostgreSQL](https://www.postgresql.org/) to store location data, food items, availability times, and other relevant information. PostgreSQL's robustness and reliability made it an ideal choice for handling the data-intensive operations of our application.

### Caching the Map for Performance

One of the critical features of *Free at UCD* is the interactive map that displays food giveaway locations. Initially, users experienced delays and reloads when zooming or panning the map. To enhance performance, we implemented caching mechanisms:

- **Map Caching:** By caching the map object, we reduced the load times significantly. Users can now navigate the map smoothly without unnecessary reloads.
- **Data Caching:** We also cached database queries to minimize the number of calls to our PostgreSQL database, thereby improving the application's responsiveness.

### Deployment on Render

We deployed the application on [Render](https://render.com/), a cloud platform known for its ease of use and scalability. Despite being a large-scale application, we managed to run it efficiently on a **bare minimum CPU configuration**, thanks to our optimized code and caching strategies.

*Figure 1: Overview of Free at UCD's Architecture*

## Day One Success

The response to *Free at UCD* has been overwhelming. On the first day alone, we had over **100 active users** interacting with the platform. The feedback has been incredibly positive, with students appreciating the real-time updates and the ease of finding free food on campus.

*Figure 2: User Growth on Launch Day*

## Lessons Learned and Future Plans

Developing *Free at UCD* taught us valuable lessons about building scalable applications with limited resources:

- **Optimization is Key:** Efficient coding practices and caching can significantly improve performance without the need for expensive infrastructure.
- **User-Centric Design:** Focusing on the needs of the users ensured that we built a tool that provided real value.

### Upcoming Features

We are excited to roll out new features in the coming weeks:

- **Notification System:** Users will receive alerts when new food giveaways are added near their location.
- **Event Submission:** Allowing event organizers to add their own food giveaway events to the platform.
- **Enhanced Filtering:** Users will be able to filter giveaways based on dietary preferences like vegan or gluten-free options.

## Conclusion

*Free at UCD* is more than just an app; it's a community resource aimed at enhancing student life on campus. By leveraging modern technologies and focusing on performance optimization, we've created a platform that not only serves its purpose effectively but also scales efficiently.

We are grateful for the support from the UCD community and are committed to continuous improvement. Stay tuned for more updates!

## Get Involved

If you're interested in contributing to the project or have suggestions, feel free to reach out or check out our [GitHub repository](https://github.com/Yash-Singh-Pathania/free_at_ucd).

---

*Thank you for being part of our journey!*

